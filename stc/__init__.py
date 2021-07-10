# -*- coding: utf-8 -*-

import sys
import csv
import sqlite3
import pandas as pd
import numpy as np
from warnings import warn
from hashlib import md5
from pickle import dumps, loads
from time import time
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from collections import Counter
from uuid import uuid1
from typing import Union, Any, List, Tuple, FrozenSet


class SparseTensorClassifier:
    """
    This class implements the Sparse Tensor Classifier (STC), a supervised classification algorithm for categorical
    data inspired by the notion of superposition of states in quantum physics.

    The algorithm is implemented in SQL. By default, the library uses an in-memory SQLite database, shipped with
    Python standard library, that require no configuration by the user. It is also possible to configure STC to
    run on `alternative DBMS <https://github.com/SparseTensorClassifier/tutorial/blob/main/Quickstart_DBMS.ipynb>`_
    in order to take advantage of persistent storage and scalability.

    The input data must be a pandas ``DataFrame`` or a JSON structured as follows:

    .. code-block:: python

        data = [
            {'key1': [value1, value2, ..., valueN], 'key2': [], ..., 'keyN': []},
            ...
            {'key1': [value1, value2, ..., valueN], 'key2': [], ..., 'keyN': []},
        ]

    Such that each dictionary represents an item where each ``key`` is a feature associated to one or more ``values``.
    This makes easy to deal with multi-valued attributes. STC also supports input data in the form of pandas
    ``DataFrame`` for tabular data, where each row represents an item, each column represents a ``key`` and each cell
    represents a ``value``. STC deals with **categorical data** only and all the ``values`` are internally
    converted to strings. Continuous features should be discretized first.

    :param targets: The target variable(s). In the notation above, this is the list of ``keys`` to predict.
    :param features: In the notation above, this is the list of ``keys`` to use for prediction.
    :param collapse: If ``True`` (the default) merges all the ``features`` into a unique ``key`` and STC reduces to a
                     matrix-based approach. This is fast and efficient, and recommended for tabular data.
                     When ``False``, the items are represented with the cartesian product among
                     the ``values`` in each ``key``. In this case, it is needed a policy to avoid degenerate probability
                     estimates in the prediction phase. The policy can be arbitrarily specified or automatically learnt
                     with :meth:`stc.SparseTensorClassifier.learn`
    :param engine: Connection to database in the form of a
                   `SQLAlchemy engine <https://docs.sqlalchemy.org/en/13/core/engines.html>`_.
                   By default, STC uses an in-memory SQLite database.
    :param prefix: Prefix to use in the database tables. STC instances initialized with different prefix are completely
                   independent. This makes possible to use the same ``engine`` multiple times with different ``prefix``,
                   without creating a new database/schema. If an instance associated with the same ``engine`` and
                   ``prefix`` is found on the database, then STC is initialized from the database.
    :param chunksize: Number of items to fit and predict per chunk. May impact the computational time.
    :param cache: If ``True`` (the default) caches fitted weights to improve performance. The cache can be cleaned with
                  :meth:`stc.SparseTensorClassifier.clean` and it is automatically cleaned each time new data are fitted
                  with :meth:`stc.SparseTensorClassifier.fit`
    :param power: Hyper-parameter. Smaller values give similar weight to all the features regardless of their
                  frequency. Usually between 0 and 1.
    :param balance: Hyper-parameter. The sample is artificially balanced when setting ``balance=1``. It is not balanced
                    with ``balance=0``. For values between 0 and 1 the sample is semi-balanced, increasing the weight
                    of the less frequent classes but not enough to have a balanced sample. For values greater than 1 the
                    sample is super-balanced, where the weight of the less frequent classes is greater than the weight
                    of the most frequent classes.
    :param entropy: Hyper-parameter. Higher values lead to predictions based on less but more relevant features,
                    thus more robust to noise. Usually between 0 and 1.
    :param loss: Loss function used in :meth:`stc.SparseTensorClassifier.learn`. Use ``norm`` for Manhattan Distance
                 or ``log`` for cross-entropy (log-loss).
    :param tol: The actual predicted probabilities are replaced with the value of ``tol`` when using ``loss='log'`` and
                the actual predicted probability is zero.
    """

    def __init__(self, targets: List[str], features: List[str] = None, collapse: bool = True,
                 engine: Union[Engine, str] = "sqlite://", prefix: str = "stc",
                 chunksize: int = 100, cache: bool = True,
                 power: float = 0.5, balance: float = 1, entropy: float = 1,
                 loss: str = 'norm', tol: float = 1e-15) -> None:

        # validate
        if not targets:
            raise Exception("No target found. Set the target variable(s) with: targets=['t1','t2','...']")

        # engine
        if isinstance(engine, str):
            self.engine = create_engine(engine, echo=False)
        else:
            self.engine = engine

        # db name
        self.db = self.engine.url.get_dialect().name

        # connect
        self.conn = None
        self.connect()

        # fields
        self.name_field = "name"
        self.item_field = "item"
        self.dim_field = "dimension"
        self.value_field = "value"
        self.score_field = "score"

        # tables
        self.tmp_table = f"{prefix}_tmp"
        self.meta_table = f"{prefix}_meta"
        self.dim_table = f"{prefix}_dims"
        self.value_table = f"{prefix}_values"
        self.train_table = f"{prefix}_train"
        self.corpus_table = f"{prefix}_corpus"

        # collapse key
        self.collapse_key = "features"
        self.collapse = self._meta(key=self.collapse_key, default=None)

        # sanitize dims
        dims = None
        targets = self._sanitize(targets)
        features = self._sanitize(features)
        if features:
            if set(features).intersection(set(targets)):
                raise Exception("Features should not contain any target variable")
            if collapse:
                dims = [self.collapse_key] + targets
            else:
                dims = features + targets

        # try to read the dimensions from the DB
        try:
            x = self.read_sql(f"SELECT * FROM {self.dim_table}")
            self.dims = x[self.name_field].tolist()
            self.dims_map = dict(zip(x[self.name_field], x[self.dim_field]))

        # otherwise read from argument
        except SQLAlchemyError:
            if dims:
                self.dims = dims
                self.dims_map = dict([(d, i) for i, d in enumerate(self.dims)])
                x = pd.DataFrame([{self.name_field: k, self.dim_field: v} for k, v in self.dims_map.items()])
                self.to_sql(x, self.dim_table)
            else:
                raise Exception("No dimensions provided and no dimensions found in DB")

        # check dimensions mismatch
        if dims:
            if set(dims) != set(self.dims):
                raise Exception("Dimensions found in DB different from the dimensions provided")

        # check collapse mismatch
        if self.collapse is None:
            self.collapse = features if collapse else []
            self._meta(key=self.collapse_key, value=self.collapse)
        elif collapse and features:
            if set(self.collapse) != set(features):
                raise Exception("Dimensions found in DB different from the dimensions provided")

        # init values map from DB or from scratch
        try:
            x = self.read_sql(f"SELECT * FROM {self.value_table}")
            self.values_map = self._defaultdict(zip(x[self.name_field], x[self.value_field]))
        except SQLAlchemyError:
            self.values_map = self._defaultdict()

        # encrypt
        self.dims = self._encrypt(self.dims)
        self.targets = self._encrypt(targets)

        # set params
        self.chunksize = chunksize
        self.cache = cache
        self.power = power
        self.balance = balance
        self.entropy = entropy
        self.loss = loss
        self.tol = tol
        self.qtable = {}
        self.set({})

    """""""""""""""""""""""""""""""""""""""""""""
    PUBLIC METHODS
    """""""""""""""""""""""""""""""""""""""""""""

    def connect(self) -> None:
        """
        Open the connection to the database.

        """

        self.conn = self.engine.connect()

        # add POW & LOG support to SQLite
        if self.db == "sqlite":
            self.conn.connection.create_function("POW", 2, pow)
            self.conn.connection.create_function("LOG", 1, np.log)

    def close(self) -> None:
        """
        Close the connection to the database.

        """

        self.conn.close()

    def clean(self, deep: bool = False) -> None:
        """
        Clean the database.

        :param deep: If ``False`` (the default) drops temporary tables and cache. If ``True``, deletes all tables and
                     closes the connection.
        """

        # drop tmp tables
        tmp = self._meta(key=self.tmp_table)
        if tmp is not None:
            for table in tmp:
                self._DROP(table)

        # clear cache
        self.conn.execute(text(f"DELETE FROM {self.meta_table} WHERE {self.name_field} LIKE :like"), like='cache-%')

        # drop all tables and destroy connection
        if deep:
            for table in [self.meta_table, self.dim_table, self.value_table, self.train_table, self.corpus_table]:
                self._DROP(table)
            self.engine = None
            self.close()

    def set(self, params: dict) -> None:
        """
        Set parameters. Changing parameters does NOT need to re-fit STC. The fitting of STC is independent
        from the parameters. In particular, also the ``targets`` can be changed on the fly (if initialized
        with ``collapse=False``).

        :param params: Dictionary of parameters to set in the form of ``{'param': value}``. Supported parameters are:
                       ``targets``, ``chunksize``, ``cache``, ``power``, ``balance``, ``entropy``, ``loss``, ``tol``.

        """

        allow = {"chunksize", "cache", "targets", "power", "balance", "entropy", "loss", "tol"}
        keys = params.keys()
        extra = set(keys) - allow
        if len(extra) > 0:
            ee = ", ".join(extra)
            aa = ", ".join(allow)
            raise Exception(f'The parameters "{ee}" cannot be set. Allowed parameters: "{aa}"')

        def set_param(param, val):
            if param in ["targets"]:
                val = self._encrypt(self._sanitize(val))
            setattr(self, param, val)

        for param, val in params.items():
            set_param(param, val)

        self.qtable = self._meta(key=self._qkey(), default={})

    def get(self, params: Union[str, List[str]]) -> Any:
        """
        Get parameters. Read the parameters provided upon initialization.
        
        :param params: Name(s) of the parameters to return.
        :return: Value(s) of the parameters.
        """""

        def get_param(param):
            val = getattr(self, param)
            if param in ["dims", "targets"]:
                val = self._decrypt(val)
            return val

        if isinstance(params, str):
            return get_param(params)

        val = {}
        for param in params:
            val[param] = get_param(param)

        return val

    def fit(self, items: Union[List[dict], pd.DataFrame], keep_items: bool = None, if_exists: str = "fail",
            clean: bool = True) -> None:
        """
        Fit the training data. The data must contain both ``targets`` and ``features`` and must be structured
        as described above. Supports incremental fit and it is ready to use in an online learning context.

        :param items: The training data in JSON or tabular format as described above.
        :param keep_items: If ``True``, stores the individual items seen during fit. This requires longer computational
                           times but allows to estimate the policy with :meth:`stc.SparseTensorClassifier.learn`.
                           By default, it is ``False`` when ``collapse=True`` or when only
                           a single ``target`` and a single ``feature`` have been provided upon initialization. In this
                           case, there is no need to estimate the policy and no need to store the individual items.
        :param if_exists: The action to take if STC has already been fitted. One of ``fail``: raise exception,
                          ``append``: incremental fit in online learning, ``replace``: re-fit from scratch.
        :param clean: If ``True`` (the default) invalidates the cache used for prediction.
        """

        # clean cache
        if clean:
            self.clean()

        # keep items
        if keep_items is None:
            keep_items = False if len(self.dims) == 2 else True

        # sanitize
        items_all = self._stringify(items)

        # number of items
        n_all = len(items_all)

        # universal identifiers
        uids_all = self._uids(n=n_all, data_table=self.train_table, if_exists=if_exists)

        # progress bar
        progress = self._progress(total=n_all, status="Fitting")

        # fit chunks
        for c in range(0, n_all, self.chunksize):

            # chunk
            items = items_all[c:c + self.chunksize]
            uids = uids_all[c:c + self.chunksize]

            # add key:value pairs to the dictionary
            new = []
            for i, item in enumerate(items):

                if isinstance(item, dict):
                    dv = item.items()
                else:
                    dv = [(item, items.iloc[:, i].values)]

                for dim, values in dv:
                    try:
                        for v in set(values):
                            n = len(self.values_map)
                            m = self.values_map.setdefault(v, n)
                            if n == m:
                                new.append({self.name_field: v, self.value_field: m})
                    except KeyError:
                        pass

            # update the dictionary
            if new:
                x = pd.DataFrame(new)
                self.to_sql(x, self.value_table, if_exists="append")

            # transform
            self._transform(items=items, uids=uids, dims=self.dims,
                            data_table=self.train_table if keep_items else None,
                            corpus_table=self.corpus_table, if_exists=if_exists)

            # append chunks
            if_exists = "append"

            # update progressbar
            progress.update(c + self.chunksize)

    def explain(self, features: List[str] = None) -> pd.DataFrame:
        """
        Global explainability. Compute the global contribution of each feature value to each target class label.

        :param features: The features to use. By default, it uses the ``features`` used for prediction.
        :return: Global explainability table giving the contribution of each feature value to each target class label.
        """

        # features
        if features is not None:
            features = self._encrypt(self._sanitize(features))
        else:
            features = self._difflst(self._encrypt(self._flatten(self.policy()[0])), self.targets)

        # weights
        if self.cache:
            sql = self._SELECT_cache(features=features, corpus_table=self.corpus_table)
        else:
            sql = self._SELECT_robust(features=features, corpus_table=self.corpus_table)
            sql = f"WITH {sql[1]} {sql[0]}"

        # query
        t = self._decrypt(self.targets)
        w = self._decrypt(self.read_sql(sql))
        w.sort_values(t + [self.score_field], ascending=[True for _ in self.targets] + [False], inplace=True)

        # return
        return w.set_index(t)

    def predict(self, items: Union[List[dict], pd.DataFrame], policy: List[List[str]] = None, probability: bool = True,
                explain: bool = True) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
        """
        Predict the test data. The data must be structured as described above and must contain the ``features``.
        All additional keys are ignored (included ``targets``).
        If all the attributes of an item are associated with features never seen in the training set, STC will not be
        able to provide a prediction. In this case, a fallback mechanism is needed: the ``policy``. The ``policy`` is
        a list of sets of features to use subsequently for prediction.
        The algorithm starts with the first element of the policy (first set of features).
        If no prediction could be produced, the second set of features is used, and so on.
        If the policy ends with the empy list ``[]``, then all the item are guaranteed to be predicted
        (eventually using no features, i.e., they will be attributed the most likely class in the trainig set).
        If the policy does not end with the empy list ``[]``, some predictions may miss for some items.

        :param items: The test data in JSON or tabular format as described above.
        :param policy: List of lists of features to use for prediction, such as ``[[f1,f2],[f1],[]]``.
                       First lists are applied first. By default, it uses the policy ``[[features],[]]`` when
                       only one feature is provided upon initialization or when ``collapse=True``. In the other cases,
                       it uses the policy ``[[features]]`` and raises a warning as some predictions may miss for some
                       items. If a policy has been learnt with :meth:`stc.SparseTensorClassifier.learn`, it uses that
                       policy instead.
        :param probability: If ``True`` (the default) returns the probability of the target class label for
                            each predicted item. If ``False``, and also ``explain=False``, returns the final
                            classification only (saves computational time).
        :param explain: If ``True`` (the default) returns the contribution of each feature to the target class label
                        for each predicted item.
        :return: Tuple of (classification, probability, explainability). The classification table contains the final
                 predictions for each item. Missing predictions are encoded as ``NaN``. The probability table contains
                 the probabilities of the target class labels for each predicted item. Labels that do not appear in
                 this table are associated with zero probability. The explainability table provides the contribution of
                 each feature to the target class label for each predicted item.
        """

        # sanitize
        items_all = self._stringify(items)

        # number of items
        n_all = len(items_all)

        # universal identifiers
        uids_all = self._uids(n=n_all)

        # policy
        if policy is None:
            policy = self.policy()[0]

        # check policy
        if policy[-1]:
            warn("Bad policy."
                 " The last element of the policy is not empty and predictions may miss for some items."
                 " Use the method 'learn()' to learn the optimal policy or specify a custom policy ending with"
                 " the empty list [].")

        # features
        features = self._encrypt(self._flatten(policy))

        # progress bar
        progress = self._progress(total=n_all, status="Predicting")

        # fit chunks
        x = None
        for c in range(0, n_all, self.chunksize):

            # chunk
            items = items_all[c:c + self.chunksize]
            uids = uids_all[c:c + self.chunksize]

            # tmp table
            tmp_test = self._tmp_table()

            # transform data
            self._transform(items=items, uids=uids, dims=features, data_table=tmp_test)

            # predict
            miss = None
            for p in policy:

                # features
                p = self._encrypt(self._sanitize(p))

                # predict query
                sql_select, sql_with = self._SELECT_predict(
                    features=p, corpus_table=self.corpus_table, data_table=tmp_test,
                    ids=miss, probability=probability, explain=explain, cache=self.cache)
                sql = f"WITH {sql_with} {sql_select}"

                # append to explanatory table
                if x is None:
                    x = self.read_sql(sql)
                else:
                    x = x.append(self.read_sql(sql))

                # missing predictions
                miss = ",".join(map(str, set(uids) - set(x[self.item_field].values)))
                if not miss:
                    break

            # clean
            self._DROP(tmp_test)

            # update progressbar
            progress.update(c + self.chunksize)

        # back to original dims and values
        t = self._decrypt(self.targets)
        x = self._decrypt(x)

        # explainability -> probability
        e, p = None, None
        if explain:
            # explainability
            e = x.set_index([self.item_field] + t).sort_values(
                [self.item_field] + t + [self.score_field], ascending=[True] + [True for _ in t] + [False])
            # probability
            p = e.groupby([self.item_field] + t).agg({self.score_field: 'sum'})
            p[self.score_field] = pow(p[self.score_field], 1. / self.power)
            p[self.score_field] = p[self.score_field] / p.groupby(self.item_field)[self.score_field].transform('sum')
            p.reset_index(inplace=True)
        elif probability:
            # probability
            p = x

        # probability -> classification
        if p is not None:
            # probability
            p = p.pivot(index=self.item_field, columns=t, values=self.score_field).fillna(0).sort_index()
            p = p.reindex(sorted(p.columns), axis=1)
            # classification
            c = p.idxmax(axis=1)
            c = pd.DataFrame([i for i in c.values], columns=t, index=c.index)
        else:
            # classification
            c = x.set_index(self.item_field)

        # fill missing predictions
        miss = set(uids_all) - set(c.index)
        if miss:
            n = range(len(t))
            for m in miss:
                c.loc[m] = [np.nan for _ in n]

        # sort index
        c.sort_index(inplace=True)

        # return
        return c, p, e

    def learn(self, test_size: float = None, train_size: float = None, stratify: bool = True,
              priority: List[str] = None, max_features: int = 0, max_actions: int = 0,
              max_iter: int = 1, max_runtime: int = 0, random_state: bool = None) \
            -> Tuple[List[List[str]], List[float]]:
        """
        Learn the ``policy``. Learn the ``policy`` via reinforcement learning by optimizing the ``loss`` function
        on cross validation. Before learning, STC must be fitted with ``keep_items=True``. Then, proceeds as follows.
        For each episode, split the train set in train-validation sets.
        Start with a set of empy features and compute the reward of the state (-loss).
        Add the value to the Q-table. Explore all the next states generated by adding 1 feature to the empty set.
        Compute the values of the states. Add to the Q-table. Select the state with the maximum value.
        Move to that state. Explore all the next states generated by adding 1 feature to the current set of features...
        Stop when all features are used or when the value of all the next states is less than the value of the current
        state.

        :param test_size: Train-test cross validation split (percentage of the training sample).
        :param train_size: Train-test cross validation split (percentage of the training sample).
        :param stratify: If ``True``, the folds are made by preserving the percentage of samples for each class.
        :param priority: List of features to learn first.
        :param max_features: Number of maximum features to return in the policy. If 0, no limit.
        :param max_actions: Number of maximum states to explore at once. If 0, no limit.
        :param max_iter: Maximum number of iterations to train the algorithm. If 0, no limit.
        :param max_runtime: How long to train the algorithm, in seconds. If 0, no time limit.
        :param random_state: Random number generator seed, used for reproducible output across function calls.
        :return: Tuple of (policy, loss). The policy is saved internally and used by default in
                 :meth:`stc.SparseTensorClassifier.predict`. The second element of the tuple provides the loss
                 associated with the policy.
        """

        # validate
        if max_iter <= 0 and max_runtime <= 0:
            raise Exception("Cannot set both 'max_iter' and 'max_runtime' to 0. The algorithm would run endlessly.")
        if max_runtime > 0 and random_state is not None:
            raise Exception("Cannot set 'random_state' with runtime limit. This would lead to undesired results on "
                            "machines with different computational resources. Set 'max_runtime=0' and use 'max_iter' "
                            "instead.")

        # features
        features = self._difflst(self.dims, self.targets)

        # priority
        if priority is not None:
            priority = self._encrypt(self._sanitize(priority))

        # items
        t = self._COLS(self.train_table, self.targets)
        items = self.read_sql(
            f"SELECT DISTINCT {self.item_field}, {t} FROM {self.train_table} ORDER BY {self.item_field}, {t}")

        # start countdown
        tic = time()
        toc = time()
        iteration = 0
        while (toc-tic < max_runtime or max_runtime <= 0) and (iteration < max_iter or max_iter <= 0):

            # train-test split
            train, test = self._train_test_split(
                items, test_size=test_size, train_size=train_size, stratify=stratify, random_state=random_state)

            # unset random_state
            random_state = None

            # convert to ids
            ids_train = ",".join(map(str, train))
            ids_test = ",".join(map(str, test))

            # number of test items
            n = len(test)

            # fallback and corpus tables
            tmp_state = None
            tmp_corpus = self._tmp_table()

            # create the corpus from the train set
            sql_select = self._SELECT_marginal(dims=self.dims, table=self.train_table, ids=ids_train, corpus=True)
            self._CREATE(table=tmp_corpus, sql_select=sql_select)

            # start with an empty policy
            state = []
            actions = [[]]
            while toc-tic < max_runtime or max_runtime <= 0:

                # progress bar
                progress = self._progress(
                    total=len(actions), status=f"Learning iteration {iteration+1} state {len(state)}")

                # explore all actions
                for a in actions:

                    # compute the distance in the next state. Use current distances as fallback
                    loss = self._SELECT_loss(
                        features=a+state[0] if state else a, corpus_table=tmp_corpus,
                        data_table=self.train_table, fallback_table=tmp_state, ids=ids_test)

                    # compute the reward
                    reward = self.conn.execute((
                        f"WITH {loss[1]}, loss AS ({loss[0]}) "
                        f"SELECT -SUM({self.score_field}) AS {self.score_field} "
                        f"FROM loss"
                    )).scalar()

                    # add the reward to the Q-table
                    key = self._hash(self._action(state, a))
                    val = np.array([n, reward], dtype=np.float64)
                    if key in self.qtable:
                        self.qtable[key] += val
                    else:
                        self.qtable[key] = val

                    # update progressbar
                    progress.step()

                    # check runtime
                    toc = time()
                    if toc-tic > max_runtime > 0:
                        break

                # select the best action and move to the next state
                state, action, value = self._next(state=state, actions=actions)
                if action is None or 0 < max_features <= len(state[0]):
                    break

                # generate actions for the state
                actions = self._actions(state=state, priority=priority, max_actions=max_actions)
                if actions is None:
                    break

                # compute the loss in the state
                loss = self._SELECT_loss(
                    features=state[0], corpus_table=tmp_corpus, data_table=self.train_table,
                    fallback_table=tmp_state, ids=ids_test)

                # store the distances
                tmp_fallback = self._tmp_table()
                self._CREATE(table=tmp_fallback, sql_with=loss[1], sql_select=loss[0])

                # drop previous loss
                tmp_fallback, tmp_state = tmp_state, tmp_fallback
                if tmp_fallback is not None:
                    self._DROP(tmp_fallback)

                # check runtime
                toc = time()

            # clean DB
            self._DROP(tmp_state)
            self._DROP(tmp_corpus)

            # update runtime
            toc = time()
            iteration += 1

        # store the Q-table in the DB
        self._meta(key=self._qkey(), value=self.qtable)

        # return the best policy
        return self.policy()

    def policy(self) -> Tuple[List[List[str]], List[float]]:
        """
        Get the ``policy``.

        :return: Output of :meth:`stc.SparseTensorClassifier.learn`
        """

        # init state and values
        state, values = [], []

        # generate policy
        while True:
            state, action, value = self._next(state=state)
            if action is None:
                break
            else:
                values = [value] + values

        # fallback if no policy: use all features
        if not state:
            features = self._difflst(self.dims, self.targets)
            state = [features]
            if len(features) == 1:
                state += [[]]

        # map policy to original dimensions
        for i in range(len(state)):
            state[i] = self._decrypt(state[i])

        # return policy and values
        return state, values

    def read_sql(self, sql: str) -> pd.DataFrame:
        """
        Read SQL query into a pandas ``DataFrame``.

        :param sql: SQL query to SELECT data.
        :return: Output of the SQL query.
        """

        return pd.read_sql(sql, self.conn)

    def to_sql(self, x: pd.DataFrame, table: str, if_exists: str = 'fail') -> None:
        """
        Write a pandas ``DataFrame`` into a SQL table.

        :param x: A pandas ``DataFrame`` to write into ``table``.
        :param table: The name of the table to write ``x`` into.
        :param if_exists: The action to take when the table already exists. One of ``fail``: raise exception,
                          ``append``: insert new values to the existing table, ``replace``: drop the table before
                          inserting new values.
        """

        def csv_insert(table, conn, keys, data_iter):
            x = pd.DataFrame(list(data_iter))
            c = ",".join(keys)
            v = x.to_csv(line_terminator="),(", header=False, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
            sql = f"INSERT INTO {table.name} ({c}) VALUES({v})"
            conn.execute(sql[:-3])

        if self.db == "oracle":
            def csv_insert(table, conn, keys, data_iter):
                x = pd.DataFrame(list(data_iter))
                c = ",".join(keys)
                t = f") INTO {table.name} ({c}) VALUES ("
                v = x.to_csv(line_terminator=t, header=False, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
                sql = f"INSERT ALL INTO {table.name} ({c}) VALUES({v})"
                sql = sql[:-len(t)]
                sql += "SELECT * FROM DUAL"
                conn.execute(sql)

        # SQL Server requires chunksize=1000: maximum allowed rows in bulk insert.
        x.to_sql(table, self.conn, if_exists=if_exists, index=False, method=csv_insert, chunksize=1000)

    """""""""""""""""""""""""""""""""""""""""""""
    PRIVATE METHODS
    """""""""""""""""""""""""""""""""""""""""""""

    def _encrypt(self, x: Union[List[str], pd.DataFrame]) -> Union[List[str], pd.DataFrame]:
        """Encryption"""

        if isinstance(x, list):
            return ['d' + str(self.dims_map[d]) for d in x]

        y = pd.DataFrame()
        for d in x.columns:
            if d in self.dims_map:
                i = self.dims_map[d]
                y['d' + str(i)] = x[d].map(self.values_map)
            else:
                y[d] = x[d]

        return y

    def _decrypt(self, x: Union[List[str], pd.DataFrame]) -> Union[List[str], pd.DataFrame]:
        """Decryption"""

        dinv = {'d' + str(d): k for k, d in self.dims_map.items()}

        if isinstance(x, list):
            return [dinv[d] for d in x]

        vinv = {v: k for k, v in self.values_map.items()}

        y = pd.DataFrame()
        for i in x.columns:
            if i in dinv:
                d = dinv[i]
                y[d] = x[i].map(vinv)
            else:
                y[i] = x[i]

        return y

    def _flatten(self, policy: List[List[str]]) -> List[str]:
        """Convert a list of lists to a list"""

        if not all(isinstance(dims, list) for dims in policy):
            raise Exception("Invalid policy. Policy must be a list of lists: [[],[],...,[]]")

        return self._sanitize([dim for dims in policy for dim in dims])

    def _sanitize(self, dims: List) -> List[str]:
        """Return list of strings with unique elements in the same order"""

        if dims is None:
            return []
        if isinstance(dims, str):
            dims = [dims]
        if not isinstance(dims, list):
            dims = list(dims)

        return self._unique([str(d) for d in dims])

    def _unique(self, x: List) -> List:
        """Drop duplicates from list and maintain the order"""

        u = []
        for i in x:
            if i not in u:
                u.append(i)

        return u

    def _difflst(self, x: List, y: List) -> List:
        """List difference"""

        return [i for i in x if i not in y]

    def _stringify(self, items: Union[List[dict], pd.DataFrame]) -> Union[List[dict], pd.DataFrame]:
        """Convert keys and values to string"""

        # dims
        dims = self._decrypt(self.targets) + self.collapse if self.collapse else self._decrypt(self.dims)

        # json data
        if isinstance(items, list):
            json = []
            for data in items:
                item = {self.collapse_key: []} if self.collapse else {}
                for key, val in data.items():
                    key, val = str(key), val if isinstance(val, list) else [val]
                    if key in self.collapse:
                        item[self.collapse_key] += [key + ": " + str(v) for v in val]
                    elif key in dims:
                        item[key] = [str(v) for v in val]
                json.append(item)
            return json

        # tabular data
        items.columns = items.columns.astype(str)
        items = items[dims].astype(str)
        if self.collapse:
            json = []
            for _, data in items.iterrows():
                item = {self.collapse_key: []}
                for k, v in data.items():
                    if k in self.collapse:
                        item[self.collapse_key] += [k + ": " + v]
                    else:
                        item[k] = [v]
                json.append(item)
            return json

        return items

    def _ckey(self, corpus_table: str, features: List[str]) -> str:
        """Generate key for cache"""

        key = sorted(self.targets) + [corpus_table] + sorted(features) + [self.power, self.balance, self.entropy]

        return "cache-" + md5("#".join(map(str, key)).encode()).hexdigest()

    def _qkey(self) -> str:
        """Generate key for qtable"""

        key = sorted(self.targets) + [self.power, self.balance, self.entropy, self.loss, self.tol]

        return "qtable-" + md5("#".join(map(str, key)).encode()).hexdigest()

    def _hash(self, state: List[List[str]]) -> Tuple[FrozenSet[str]]:
        """Hash a state

        A state represents a policy. The order of the lists of features composing the policy does count, but the order
        of the features in each list does not. Generate the corresponding hash.

        :param state: a policy, list of lists of features
        :return: hashable object, tuple of frozensets
        """

        return tuple([frozenset(s) for s in state])

    def _action(self, state: List[List[str]], action: List[str]) -> List[List[str]]:
        """Do action and generate the new set of features

        Add the features to the last policy and return the list of features.

        :param state: a policy, list of lists of features
        :param action: list of features
        :return: list of features
        """

        return [action + state[0]] + state if state else [action]

    def _actions(self, state: List[List[str]], priority: List[str] = None, max_actions: int = 0) \
            -> Union[List[List[str]], None]:
        """Generate actions for the current state"""

        # features
        features = self._difflst(self.dims, self.targets)

        # empty action
        actions = [[]]
        if not state:
            return actions

        # priority actions
        if priority is not None:
            actions = [[a] for a in self._difflst(priority, state[0])]

        # all actions
        if priority is None or not actions:
            actions = [[a] for a in self._difflst(features, state[0])]
            if not actions:
                return None

        # top actions
        if 0 < max_actions < len(actions):
            miss = []
            rank = []
            # rank actions and identify unexplored actions
            for a in actions:
                s = self._action(state, a)
                try:
                    v = self.qtable[self._hash(s)]
                    v = v[1] / v[0]
                    rank += [(v, a)]
                except KeyError:
                    miss += [a]
            # sort best to worst
            if rank:
                rank.sort(key=lambda x: -x[0])
            # merge top and unexplored actions
            actions = miss + [x[1] for x in rank[0:max_actions]]

        # return
        return actions

    def _next(self, state: List[List[str]], actions: List[List[str]] = None) \
            -> Tuple[List[List[str]], List[str], float]:
        """Move to the next state

        Generate the next state based on the best action. The best action is found by exploiting the Q-table.

        :param state: current policy
        :param actions: possible actions from the current state
        :return: next state generated by the best action
        """

        # actions
        if actions is None:
            actions = self._actions(state=state)

        # init state, action, value
        s_, a_, v_ = state, None, None
        if self._hash(s_) in self.qtable:
            v_ = self.qtable[self._hash(s_)]
            v_ = v_[1] / v_[0]

        # best action -> next state
        if actions is not None:
            for a in actions:
                s = self._action(state, a)
                try:
                    v = self.qtable[self._hash(s)]
                    v = v[1] / v[0]
                    if v_ is None or v > v_:
                        s_, a_, v_ = s, a, v
                except KeyError:
                    pass

        # return
        return s_, a_, v_

    def _train_test_split(self, items: pd.DataFrame, test_size: float = None,
                          train_size: float = None, stratify: bool = True,
                          random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Train-Test Split

        Split data into random train and test subsets.

        :param items: the data in tabular format
        :param test_size: train-test cross validation split, percentage of train data
        :param train_size: train-test cross validation split, percentage of test data
        :param stratify: if True, the folds are made by preserving the percentage of samples for each class
        :param random_state: random number generator seed. Pass an int for reproducible output across function calls.
        :return: Tuple of arrays containing the IDs of the train-test data
        """

        if random_state is not None:
            np.random.seed(random_state)

        if train_size is None and test_size is None:
            test_size = 0.25
        if train_size is None:
            train_size = 1 - test_size
        if test_size is None:
            test_size = 1 - train_size

        if train_size < 0 or train_size > 1:
            raise Exception("train_size represents the proportion of the dataset to include in the train split.")
        if test_size < 0 or test_size > 1:
            raise Exception("test_size represents the proportion of the dataset to include in the test split.")

        if not stratify:
            items = np.unique(items[self.item_field])
            train = np.random.choice(items, size=int(len(items) * train_size), replace=False)
            items = items[~np.isin(items, train)]
            test = np.random.choice(items, size=int(len(items) * test_size / (1 - train_size)), replace=False)
        else:
            train = items.groupby(self.targets, as_index=False).apply(lambda x: x.loc[np.random.choice(
                x.index, size=int(len(x) * train_size), replace=False), :])
            items = items[~items[self.item_field].isin(train[self.item_field])]
            test = items.groupby(self.targets, as_index=False).apply(lambda x: x.loc[np.random.choice(
                x.index, size=int(len(x) * test_size / (1 - train_size)), replace=False), :])
            train = train[self.item_field].values
            test = test[self.item_field].values

        return train, test

    def _tmp_table(self) -> str:
        """Generate a unique name for temporary tables and store in DB"""

        uid = md5(str(uuid1()).encode()).hexdigest()
        tmp = self.tmp_table + "_" + uid[:16]

        meta = self._meta(key=self.tmp_table)
        meta = [tmp] if meta is None else [tmp] + meta
        self._meta(key=self.tmp_table, value=meta)

        return tmp

    def _meta(self, key: str, value: Any = None, default: Any = None) -> Any:
        """Get/Set meta values

        Meta values are stored in the DB. If no value is provided then the key is read from the DB.
        If a key:value pair is provided, the value is updated in the DB.

        :param key: meta value key
        :param value: meta value
        :param default: value to return if no key was found
        :return: meta value or default
        """

        # dump
        if value is not None:
            value = dumps(value)

        # read or update
        try:
            if value is None:
                binary = self.conn.execute(self._SELECT_meta(key=key)).scalar()
                if binary is None:
                    return default
                return loads(binary)
            else:
                self.conn.execute(text(self._UPDATE_meta(key=key)), value=value)

        # create table and insert
        except SQLAlchemyError:
            self.conn.execute(self._CREATE_meta(table=self.meta_table))
            if value is not None:
                self.conn.execute(text(self._UPDATE_meta(key=key)), value=value)

        # fallback
        return default

    def _uids(self, n: int, data_table: str = None, if_exists: str = 'fail') -> range:

        # last item id
        last = 0
        if data_table is not None and if_exists == "append":
            try:
                last = self.conn.execute(f"SELECT MAX({self.item_field})+1 from {data_table}").scalar()
            except SQLAlchemyError:
                pass

        # universal item ids
        return range(last, last + n)

    def _transform(self, items: Union[List[dict], pd.DataFrame], uids: range, dims: List[str],
                   data_table: str = None, corpus_table: str = None, if_exists: str = 'fail') -> None:
        """Transformer

        Transform the data from JSON or tabular format to DB tables.
        Generate the table containing individual items, the corpus, or both.

        :param items: the data in JSON or tabular format
        :param uids: universal identifiers
        :param dims: the dimensions to transform. Additional dimensions are ignored
        :param data_table: the name of the table to store individual items. If None, it is not generated
        :param corpus_table: the name of the table to store the corpus. If None, it is not generated
        :return: list of ids of the items transformed
        """

        # tmp tables
        tmp_items = None
        is_tmp_data = False

        # transform dictionary to data table
        if isinstance(items, list):

            # add uids to items
            items = dict(zip(uids, items))

            # all dims as set
            keys = self._decrypt(dims)

            # map the data in form of a table with 'item': i, 'dimension': k, 'value': z, 'score': w
            if dims:

                # map values
                rid = -1
                rows = []
                for i, item in items.items():
                    rid += 1
                    miss = self._difflst(keys, item.keys())
                    if miss:
                        warn("Missing value."
                             f" Item {rid} misses {', '.join(miss)} and it will be ignored."
                             f" Consider setting a string to represent the missing value(s).")
                    for dim, value_list in item.items():
                        if not value_list:
                            warn("Missing value."
                                 f" Item {rid} has no value for '{dim}' and it will be ignored."
                                 f" Consider setting a string to represent the missing value(s).")
                        elif dim in self.dims_map:
                            m = 0
                            k = self.dims_map[dim]
                            for v, n in Counter(value_list).most_common():
                                v = self.values_map[v]
                                if v == -1:
                                    m += n
                                else:
                                    # insert values
                                    rows.append({
                                        self.item_field: i, self.dim_field: k,
                                        self.value_field: v, self.score_field: n})
                            if m > 0:
                                # insert missing values
                                rows.append({
                                    self.item_field: i, self.dim_field: k,
                                    self.value_field: -1, self.score_field: m})

                # convert to data frame
                x = pd.DataFrame(rows)

                # normalize
                x[self.score_field] = x[self.score_field] / x.groupby([self.item_field, self.dim_field])[
                    self.score_field].transform('sum')

            # empty features
            else:
                x = pd.DataFrame({self.item_field: uids, self.score_field: 1.})

            # store items in temporary table
            tmp_items = self._tmp_table()
            self.to_sql(x, tmp_items)

            # update data
            if data_table is not None:
                create = self._CREATE_data(dims=dims, table=data_table)
                update = self._UPDATE_data(dims=dims, data_table=data_table, items_table=tmp_items)

                if if_exists == "append":
                    try:
                        self.conn.execute(update)
                    except SQLAlchemyError:
                        self.conn.execute(create)
                        self.conn.execute(update)

                elif if_exists == "replace":
                    self._DROP(data_table)
                    self.conn.execute(create)
                    self.conn.execute(update)

                else:
                    self.conn.execute(create)
                    self.conn.execute(update)

        # transform DataFrame to data table
        elif isinstance(items, pd.DataFrame):

            # map data
            x = pd.DataFrame({self.item_field: uids, self.score_field: 1.})
            if dims:
                items = self._encrypt(items.reset_index(drop=True))
                x = pd.concat([x, items[dims]], axis=1)

            # switch data table
            if data_table is None:
                data_table = self._tmp_table()
                is_tmp_data = True

            # store data
            self.to_sql(x, data_table, if_exists=if_exists)

        # transform not supported
        else:
            raise Exception("Data format not supported. Use List[dict] for JSON or DataFrame for tabular data")

        # update corpus
        if corpus_table is not None:
            create = self._CREATE_corpus(dims=dims, table=corpus_table)
            update = self._UPDATE_corpus(dims=dims, corpus_table=corpus_table,
                                         data_table=data_table, items_table=tmp_items,
                                         ids=",".join(map(str, uids)))

            if if_exists == "append":
                try:
                    self.conn.execute(update)
                except SQLAlchemyError:
                    self.conn.execute(create)
                    self.conn.execute(update)

            elif if_exists == "replace":
                self._DROP(corpus_table)
                self.conn.execute(create)
                self.conn.execute(update)

            else:
                self.conn.execute(create)
                self.conn.execute(update)

        # clean
        if tmp_items is not None:
            self._DROP(tmp_items)
        if is_tmp_data:
            self._DROP(data_table)

    def _AS(self) -> str:
        """SQL AS table alias"""

        if self.db == "oracle":
            return ""

        return "AS"

    def _POW(self) -> str:
        """SQL function POW"""

        if self.db == "mssql" or self.db == "oracle":
            return "POWER"

        return "POW"

    def _LOG(self) -> str:
        """SQL function natural logarithm"""

        if self.db == "oracle":
            return "LN"

        return "LOG"

    def _COLS(self, table: str, cols: Union[str, List[str]]) -> str:
        """Formatting fields for SQL clauses"""

        if isinstance(cols, str):
            cols = [cols]

        return ",".join([f"{table}.{i}" for i in cols])

    def _ON_NATURAL(self, x: str, y: str, on: List[str]) -> str:
        """ON clause to mimic NATURAL JOIN"""

        if not on:
            return "ON (1=1)"

        return "ON ({})".format(" AND ".join([f"{x}.{i}={y}.{i}" for i in on]))

    def _DROP(self, table: str) -> None:
        """Drop SQL table"""

        # try
        try:
            # drop table
            self.conn.execute(f"DROP TABLE {table}")
            # drop from tmp tables
            meta = self._meta(key=self.tmp_table)
            if meta is not None:
                self._meta(key=self.tmp_table, value=self._difflst(meta, [table]))
        # pass
        except SQLAlchemyError:
            pass

    def _CREATE(self, table: str, sql_select: str, sql_with: str = None):
        """Create SQL table from SELECT statement"""

        if self.db == "mssql":
            if sql_with is not None:
                sql = f"WITH {sql_with} SELECT * INTO {table} FROM ({sql_select}) AS {table}_"
            else:
                sql = f"SELECT * INTO {table} FROM ({sql_select}) AS {table}_"
        else:
            if sql_with is not None:
                sql = f"CREATE TABLE {table} AS WITH {sql_with} {sql_select}"
            else:
                sql = f"CREATE TABLE {table} AS {sql_select}"

        return self.conn.execute(sql)

    def _CREATE_meta(self, table: str) -> str:
        """SQL to create the meta table"""

        if self.db == "sqlite":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{self.name_field} VARCHAR(255) PRIMARY KEY, "
                f"{self.value_field} BLOB "
                f")")
        elif self.db == "postgresql":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{self.name_field} VARCHAR(255) PRIMARY KEY, "
                f"{self.value_field} BYTEA "
                f")")
        elif self.db == "mysql":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{self.name_field} VARCHAR(255) PRIMARY KEY, "
                f"{self.value_field} LONGBLOB "
                f")")
        elif self.db == "mssql":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{self.name_field} VARCHAR(255) PRIMARY KEY, "
                f"{self.value_field} VARBINARY(MAX) "
                f")")
        elif self.db == "oracle":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{self.name_field} VARCHAR2(255) PRIMARY KEY, "
                f"{self.value_field} BLOB "
                f")")
        else:
            raise Exception(f"DB {self.db} is not supported")

        return sql

    def _CREATE_data(self, dims: List[str], table: str) -> str:
        """SQL to create the data table"""

        sql = (
            f"CREATE TABLE {table} ( "
            f"{self.item_field} INTEGER, "
            f"{' '.join([f'{d} INTEGER,' for d in dims])}"
            f"{self.score_field} FLOAT "
            f")")

        return sql

    def _CREATE_corpus(self, dims: List[str], table: str) -> str:
        """SQL to create the corpus table"""

        d = ",".join(dims)
        d_integer = ", ".join([f"{d} INTEGER" for d in dims])

        if self.db == "sqlite":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{d_integer}, "
                f"{self.score_field} FLOAT, "
                f"PRIMARY KEY ({d}))")
        elif self.db == "postgresql":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{d_integer}, "
                f"{self.score_field} FLOAT, "
                f"PRIMARY KEY ({d}))")
        elif self.db == "mysql":
            sql = (
                f"CREATE TABLE {table} ( "
                f"id BINARY(16) AS (UNHEX(MD5(CONCAT_WS('#',{d})))) STORED PRIMARY KEY, "
                f"{d_integer}, "
                f"{self.score_field} FLOAT)")
        elif self.db == "mssql":
            sql = (
                f"CREATE TABLE {table} ( "
                f"id INTEGER IDENTITY(1,1) PRIMARY KEY, "
                f"{d_integer}, "
                f"{self.score_field} FLOAT)")
        elif self.db == "oracle":
            sql = (
                f"CREATE TABLE {table} ( "
                f"{d_integer}, "
                f"{self.score_field} FLOAT, "
                f"PRIMARY KEY ({d}))")
        else:
            raise Exception(f"DB {self.db} is not supported")

        return sql

    def _UPDATE_meta(self, key: str) -> str:
        """SQL to update the meta table"""

        if self.db == "sqlite":
            sql = (
                f"INSERT OR REPLACE INTO {self.meta_table} ({self.name_field}, {self.value_field}) "
                f"VALUES ('{key}', :value)")
        elif self.db == "postgresql":
            sql = (
                f"INSERT INTO {self.meta_table} ({self.name_field}, {self.value_field}) "
                f"VALUES ('{key}', :value) "
                f"ON CONFLICT ({self.name_field}) DO UPDATE SET {self.value_field}=excluded.{self.value_field}")
        elif self.db == "mysql":
            sql = (
                f"INSERT INTO {self.meta_table} ({self.name_field}, {self.value_field}) "
                f"VALUES ('{key}', :value) "
                f"ON DUPLICATE KEY UPDATE {self.value_field}=VALUES({self.value_field})")
        elif self.db == "mssql":
            sql = (
                f"IF NOT EXISTS (SELECT * FROM {self.meta_table} WHERE {self.name_field} = '{key}') "
                f"INSERT INTO {self.meta_table}({self.name_field}, {self.value_field}) "
                f"VALUES('{key}', :value) "
                f"ELSE "
                f"UPDATE {self.meta_table} "
                f"SET {self.value_field} = :value "
                f"WHERE {self.name_field} = '{key}'")
        elif self.db == "oracle":
            sql = (
                f"BEGIN "
                f"INSERT INTO {self.meta_table} ({self.name_field}, {self.value_field}) "
                f"VALUES ('{key}', :value); "
                f"EXCEPTION "
                f"WHEN DUP_VAL_ON_INDEX THEN "
                f"UPDATE {self.meta_table} "
                f"SET {self.value_field} = :value "
                f"WHERE {self.name_field} = '{key}'; "
                f"END;")
        else:
            raise Exception(f"DB {self.db} is not supported")

        return sql

    def _UPDATE_data(self, dims: List[str], data_table: str, items_table: str) -> str:
        """SQL to update the data table"""

        data = self._SELECT_data(dims=dims, items_table=items_table)
        cols = ",".join([self.item_field] + dims + [self.score_field])
        sql = f"INSERT INTO {data_table} ({cols}) {data}"

        return sql

    def _UPDATE_corpus(self, dims: List[str], corpus_table: str, data_table: str = None, items_table: str = None,
                       ids: str = None) -> str:
        """SQL to update the corpus table"""

        d = ",".join(dims)
        s = self.score_field

        if data_table is not None:
            data = self._SELECT_marginal(dims=dims, table=data_table, ids=ids, corpus=True)
        else:
            data = self._SELECT_data(dims=dims, items_table=items_table, corpus=True)

        if self.db == "sqlite":
            if sqlite3.sqlite_version_info < (3, 24, 0):
                sql = (
                    f"WITH corpus_ AS ({data}) "
                    f"INSERT OR REPLACE INTO {corpus_table} ({d}, {s}) "
                    f"SELECT {self._COLS('corpus_', dims)}, corpus_.{s}+COALESCE({corpus_table}.{s}, 0) AS {s} "
                    f"FROM corpus_ LEFT JOIN {corpus_table} {self._ON_NATURAL('corpus_', corpus_table, dims)}")
            else:
                sql = (
                    f"WITH corpus_ AS ({data}) "
                    f"INSERT INTO {corpus_table} ({d}, {s}) "
                    f"SELECT * FROM corpus_ WHERE true "
                    f"ON CONFLICT ({d}) "
                    f"DO UPDATE SET {s}={corpus_table}.{s}+excluded.{s}")
        elif self.db == "postgresql":
            sql = (
                f"WITH corpus_ AS ({data}) "
                f"INSERT INTO {corpus_table} ({d}, {s}) "
                f"SELECT * FROM corpus_ WHERE true "
                f"ON CONFLICT ({d}) "
                f"DO UPDATE SET {s}={corpus_table}.{s}+excluded.{s}")
        elif self.db == "mysql":
            sql = (
                f"INSERT INTO {corpus_table} ({d}, {s}) "
                f"SELECT * FROM ({data}) AS excluded "
                f"ON DUPLICATE KEY UPDATE {s}={corpus_table}.{s}+excluded.{s}")
        elif self.db == "mssql":
            sql = (
                f"MERGE {corpus_table} USING ({data}) AS excluded "
                f"{self._ON_NATURAL(corpus_table, 'excluded', dims)} "
                f"WHEN MATCHED THEN UPDATE SET {s}={corpus_table}.{s}+excluded.{s} "
                f"WHEN NOT MATCHED THEN INSERT ({d},{s}) VALUES ({self._COLS('excluded', dims)}, excluded.{s});")
        elif self.db == "oracle":
            sql = (
                f"MERGE INTO {corpus_table} USING ({data}) excluded "
                f"{self._ON_NATURAL(corpus_table, 'excluded', dims)} "
                f"WHEN MATCHED THEN UPDATE SET {s}={corpus_table}.{s}+excluded.{s} "
                f"WHEN NOT MATCHED THEN INSERT ({d},{s}) VALUES ({self._COLS('excluded', dims)}, excluded.{s})")
        else:
            raise Exception(f"DB {self.db} is not supported")

        return sql

    def _SELECT_meta(self, key: str) -> str:
        """SQL to select values from the meta table"""

        return f"SELECT {self.value_field} FROM {self.meta_table} WHERE {self.name_field}='{key}'"

    def _SELECT_data(self, dims: List[str], items_table: str, corpus: bool = False) -> str:
        """SQL to select the data as tensors

        Generate the SQL to select the data as tensors.
        Read the values from a table of items and generate the tensor representation.

        :param dims: the dimensions of the tensors
        :param items_table: the table of items in the form 'item': i, 'dimension': k, 'value': z, 'score': w
        :param corpus: select the corpus or the table containing individual items
        :return: SQL query
        """

        if not dims:
            return f"SELECT DISTINCT {self.item_field}, 1 AS {self.score_field} FROM {items_table}"

        i = [d[1:] for d in dims]

        s1 = f"I{i[0]}.{self.item_field} AS {self.item_field}"
        s2 = ", ".join([f"I{j}.{self.value_field} AS d{j}" for j in i])
        s3 = " * ".join([f"I{j}.{self.score_field}" for j in i])

        f = f"{items_table} {self._AS()} I{i[0]}"
        if len(i) > 1:
            f += " JOIN " + " JOIN ".join([
                f"{items_table} {self._AS()} I{j} ON I{i[0]}.{self.item_field} = I{j}.{self.item_field}"
                for j in i[1:]])

        w = " AND ".join([f"I{j}.{self.dim_field} = {j}" for j in i])

        if not corpus:
            sql = f"SELECT {s1}, {s2}, {s3} AS {self.score_field} FROM {f} WHERE {w}"
        else:
            sql = f"SELECT {s2}, SUM({s3}) AS {self.score_field} FROM {f} WHERE {w} GROUP BY {','.join(dims)}"

        return sql

    def _SELECT_marginal(self, dims: List[str], table: str, ids: str = None, corpus: bool = False) -> str:
        """SQL to select the marginal probability

        Generate the SQL to select the marginal probability from a table of individual items or a corpus.

        :param dims: the dimensions of the marginals
        :param table: table of tensor(s) to compute the marginals
        :param ids: filter by id and select only a subset of items
        :param corpus: generate the marginal for the corpus or for each individual item
        :return: SQL query
        """

        s = [f"SUM({self.score_field}) AS {self.score_field}"]
        if dims:
            s.insert(0, ",".join(dims))
        if not corpus:
            s.insert(0, self.item_field)

        sql = f"SELECT {', '.join(s)} FROM {table}"
        if ids is not None:
            sql += f" WHERE {self.item_field} IN ({ids})"
        if len(s) > 1:
            sql += f" GROUP BY {', '.join(s[:-1])}"

        return sql

    def _SELECT_norm(self, dims: List[str]) -> str:
        """SQL to select normalization factors"""

        if not dims:
            sql_select = (
                f"SELECT SUM({self.score_field}) AS {self.score_field} "
                f"FROM corpus")
        else:
            d = self._COLS("corpus", dims)
            sql_select = (
                f"SELECT {d}, SUM({self.score_field}) AS {self.score_field} "
                f"FROM corpus "
                f"GROUP BY {d}")

        return sql_select

    def _SELECT_weight(self, features: List[str], corpus_table: str) -> Tuple[str, str]:
        """SQL to select weights"""

        corpus = self._SELECT_marginal(dims=self._unique(self.targets + features), table=corpus_table, corpus=True)
        norm_f = self._SELECT_norm(dims=features)
        norm_t = self._SELECT_norm(dims=self.targets)

        sql_with = f"corpus AS ({corpus})"
        if self.balance != 0:
            sql_with += f", norm_t AS ({norm_t})"
        if self.balance != 1:
            sql_with += f", norm_f AS ({norm_f})"

        d = self._COLS("corpus", self._unique(self.targets + features))
        if self.balance == 0:
            sql_select = (
                f"SELECT {d}, "
                f"{self._POW()}(corpus.{self.score_field}/norm_f.{self.score_field}, {self.power}) "
                f"AS {self.score_field} "
                f"FROM corpus JOIN norm_f {self._ON_NATURAL('corpus', 'norm_f', features)}")
        elif self.balance == 1:
            sql_select = (
                f"SELECT {d}, "
                f"{self._POW()}(corpus.{self.score_field}/norm_t.{self.score_field}, {self.power}) "
                f"AS {self.score_field} "
                f"FROM corpus JOIN norm_t {self._ON_NATURAL('corpus', 'norm_t', self.targets)}")
        else:
            sql_select = (
                f"SELECT {d}, {self._POW()}( "
                f" corpus.{self.score_field} / ( "
                f"  {self._POW()}(norm_f.{self.score_field}, {1-self.balance}) * "
                f"  {self._POW()}(norm_t.{self.score_field}, {self.balance}) "
                f" ), {self.power}) AS {self.score_field} "
                f"FROM corpus "
                f" JOIN norm_f {self._ON_NATURAL('corpus', 'norm_f', features)} "
                f" JOIN norm_t {self._ON_NATURAL('corpus', 'norm_t', self.targets)}")

        return sql_select, sql_with

    def _SELECT_entropy(self, features: List[str]) -> Tuple[str, str]:
        """SQL to select entropy"""

        # normalization factor (balance)
        sql = self._SELECT_norm(dims=self.targets)
        sql_with = f"entropy_balance AS ({sql})"

        # probability (non-normalized)
        d = self._COLS("corpus", self._unique(self.targets + features))
        on_t = self._ON_NATURAL("corpus", "entropy_balance", self.targets)
        sql = (
            f"SELECT {d}, "
            f"corpus.{self.score_field}/entropy_balance.{self.score_field} AS {self.score_field} "
            f"FROM corpus JOIN entropy_balance {on_t}")
        sql_with += f", entropy_prob_nn AS ({sql})"

        # normalization factor (probability)
        f = self._COLS("entropy_prob_nn", features)
        sql = (
            f"SELECT {f}, SUM({self.score_field}) AS {self.score_field} "
            f"FROM entropy_prob_nn "
            f"GROUP BY {f}")
        sql_with += f", entropy_norm AS ({sql})"

        # probability (normalized)
        d = self._COLS("entropy_prob_nn", self._unique(self.targets + features))
        on_f = self._ON_NATURAL("entropy_prob_nn", "entropy_norm", features)
        sql = (
            f"SELECT {d}, entropy_prob_nn.{self.score_field}/entropy_norm.{self.score_field} AS {self.score_field} "
            f"FROM entropy_prob_nn JOIN entropy_norm {on_f}")
        sql_with += f", entropy_prob AS ({sql})"

        # number of target states
        t = self._COLS("corpus", self.targets)
        sql = (
            f"SELECT {self._LOG()}(COUNT(*)) AS {self.score_field} "
            f"FROM (SELECT DISTINCT {t} FROM corpus) {self._AS()} states")
        sql_with += f", entropy_states AS ({sql})"

        # entropy
        f = self._COLS("entropy_prob", features)
        h = f"entropy_prob.{self.score_field} * {self._LOG()}(entropy_prob.{self.score_field})"
        sql_select = (
            f"SELECT {f}, 1 + SUM({h})/(SELECT {self.score_field} FROM entropy_states) AS {self.score_field} "
            f"FROM entropy_prob "
            f"GROUP BY {f}")

        return sql_select, sql_with

    def _SELECT_robust(self, features: List[str], corpus_table: str) -> Tuple[str, str]:
        """SQL to select robust weights"""

        sql_weight = self._SELECT_weight(features=features, corpus_table=corpus_table)
        if not self.entropy or not features:
            return sql_weight

        sql_entropy = self._SELECT_entropy(features=features)
        sql_with = f"{sql_weight[1]}, weight AS ({sql_weight[0]}), {sql_entropy[1]}, entropy AS ({sql_entropy[0]})"

        d = self._COLS("weight", self._unique(self.targets + features))
        on_f = self._ON_NATURAL("weight", "entropy", features)

        h = f"entropy.{self.score_field}"
        if self.entropy != 1:
            h = f"{self._POW()}(entropy.{self.score_field}, {self.entropy})"

        sql_select = (
            f"SELECT {d}, weight.{self.score_field} * {h} AS {self.score_field} "
            f"FROM weight JOIN entropy {on_f}")

        return sql_select, sql_with

    def _SELECT_cache(self, features: List[str], corpus_table: str) -> str:
        """SQL to select cached robust weights"""

        key = self._ckey(corpus_table=corpus_table, features=features)
        cache_table = self._meta(key=key, default=None)

        if cache_table is None:
            sql = self._SELECT_robust(features=features, corpus_table=corpus_table)
            cache_table = self._tmp_table()
            self._CREATE(table=cache_table, sql_select=sql[0], sql_with=sql[1])
            self._meta(key=key, value=cache_table)

        return f"SELECT * FROM {cache_table}"

    def _SELECT_predict(self, features: List[str], corpus_table: str, data_table: str, cache: bool,
                        ids: str = None, probability: bool = True, explain: bool = False) -> Tuple[str, str]:
        """SQL to select probability

        :param features: features to use for prediction
        :param corpus_table: table containing the corpus
        :param data_table: table containing the data to predict
        :param cache: use cache
        :param ids: filter by id and select only a subset of items
        :param explain: select addends contributing to the summation or the probability itself
        :return: SQL to select the probability (or addends)
        """

        # items
        items = self._SELECT_marginal(dims=features, table=data_table, ids=ids, corpus=False)
        sql_with = f"items AS ({items})"

        # weights
        if cache:
            robust = self._SELECT_cache(features=features, corpus_table=corpus_table)
            sql_with += f", robust AS ({robust})"
        else:
            robust = self._SELECT_robust(features=features, corpus_table=corpus_table)
            sql_with += f", {robust[1]}, robust AS ({robust[0]})"

        # on and score
        on_f = self._ON_NATURAL("robust", "items", features)
        score = f"robust.{self.score_field} * {self._POW()}(items.{self.score_field}, {self.power})"

        # query
        if explain:
            # explainability
            d = self._COLS("robust", self._unique(self.targets + features))
            sql_select = (
                f"SELECT {self.item_field}, {d}, {score} AS {self.score_field} "
                f"FROM robust JOIN items {on_f}")

        else:
            # probability (non-normalized)
            t = self._COLS("robust", self.targets)
            sql = (
                f"SELECT {self.item_field}, {t}, {self._POW()}(SUM({score}), 1./{self.power}) AS {self.score_field} "
                f"FROM robust JOIN items {on_f} "
                f"GROUP BY {self.item_field}, {t}")
            sql_with += f", prob_nn AS ({sql})"

            if probability:
                # probability norm
                sql = (
                    f"SELECT {self.item_field}, SUM({self.score_field}) AS {self.score_field} "
                    f"FROM prob_nn "
                    f"GROUP BY {self.item_field}")
                sql_with += f", prob_norm AS ({sql})"

                # probability (normalized)
                t = self._COLS("prob_nn", self.targets)
                sql_select = (
                    f"SELECT prob_nn.{self.item_field}, {t}, "
                    f"prob_nn.{self.score_field}/prob_norm.{self.score_field} AS {self.score_field} "
                    f"FROM prob_nn JOIN prob_norm ON prob_nn.{self.item_field}=prob_norm.{self.item_field}")

            else:
                # classification
                t = ",".join(self.targets)
                sql = (
                    f"SELECT {self.item_field}, {t}, "
                    f"ROW_NUMBER() OVER(PARTITION BY {self.item_field} ORDER BY {self.score_field} DESC) AS idx "
                    f"FROM prob_nn")
                sql_with += f", labels AS ({sql})"
                sql_select = f"SELECT {self.item_field}, {t} FROM labels WHERE idx = 1"

        # return
        return sql_select, sql_with

    def _SELECT_loss(self, features: List[str], corpus_table: str, data_table: str,
                     fallback_table: str = None, ids: str = None) -> Tuple[str, str]:
        """SQL to select the evaluation metric"""

        targets = self.targets

        # select the marginals of the given ids as true probability
        true = self._SELECT_marginal(dims=targets, table=data_table, ids=ids, corpus=False)

        # select the predicted probabilities
        prob = self._SELECT_predict(
            features=features, corpus_table=corpus_table, data_table=data_table, ids=ids,
            probability=True, explain=False, cache=False)

        # with clause
        sql_with = f"{prob[1]}, prob AS ({prob[0]}), prob_true AS ({true})"

        # query variables
        i = self.item_field
        s = self.score_field
        on_it = self._ON_NATURAL("prob", "prob_true", [self.item_field] + targets)

        # select clause
        if self.loss == 'norm':
            sql_select = (
                f"SELECT prob.{i}, 0.5*(SUM(ABS(prob.{s}-COALESCE(prob_true.{s},0))) + (1-SUM(prob_true.{s}))) AS {s} "
                f"FROM prob LEFT JOIN prob_true {on_it} "
                f"GROUP BY prob.{i}")
        elif self.loss == 'log':
            sql_select = (
                f"SELECT prob_true.{i}, -SUM(prob_true.{s}*{self._LOG()}(COALESCE(prob.{s},{self.tol}))) AS {s} "
                f"FROM prob_true LEFT JOIN prob {on_it} "
                f"GROUP BY prob.{i}")
        else:
            raise Exception("Loss not supported. Use 'norm' for p-norm(1) loss or 'log' for cross-entropy.")

        # add fallback for missing predictions
        if fallback_table is not None:
            sql_select += f" UNION ALL SELECT {i}, {s} FROM {fallback_table} WHERE {i} NOT IN (SELECT {i} FROM prob)"

        # return SQL
        return sql_select, sql_with

    """""""""""""""""""""""""""""""""""""""""""""
    PRIVATE CLASSES
    """""""""""""""""""""""""""""""""""""""""""""

    class _defaultdict(dict):
        """"Extend dict with default value. Do not add missing keys to dict as done in collections.defaultdict"""

        def __missing__(self, key):
            return -1

    class _progress:

        def __init__(self, total, status='', bar_len=60):
            self.total = float(total)
            self.status = status
            self.bar_len = bar_len
            self.time = time()
            self.count = 0

        def step(self):
            self.count += 1
            self.update(self.count)

        def update(self, count):
            count = min(count, self.total)
            filled_len = int(round(self.bar_len * count / self.total))
            percents = round(100.0 * count / self.total, 2)
            percents = f"{percents:.2f}" if percents < 100 else f"{percents:.1f}"
            bar = '=' * filled_len + '-' * (self.bar_len - filled_len)
            elapsed = time()-self.time
            final = elapsed/count*self.total
            sys.stdout.write(f"[{bar}] {percents}% {self.status} ({self.format(elapsed)}/{self.format(final)})\r")
            sys.stdout.flush()

        def format(self, time):
            hours, rem = divmod(time, 3600)
            minutes, seconds = divmod(rem, 60)
            return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
