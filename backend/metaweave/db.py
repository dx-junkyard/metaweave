"""Neo4j connection pool — singleton wrapper.

Usage::

    from metaweave.db import get_driver

    driver = get_driver()
    with driver.session() as session:
        session.run("MERGE (u:User {id: $id})", id="alice")

Environment variables
---------------------
NEO4J_URI   : bolt URI of the Neo4j instance  (default: bolt://neo4j:7687)
NEO4J_AUTH  : "<user>/<password>" pair         (default: neo4j/password)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from neo4j import Driver, GraphDatabase

logger = logging.getLogger(__name__)


class _Neo4jSingleton:
    """Lazily initialised, process-wide Neo4j driver (connection pool)."""

    _instance: Optional["_Neo4jSingleton"] = None
    _driver: Optional[Driver] = None

    def __new__(cls) -> "_Neo4jSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_driver(self) -> Driver:
        if self._driver is None:
            uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
            auth_env = os.environ.get("NEO4J_AUTH", "neo4j/password")
            user, _, password = auth_env.partition("/")
            logger.info("Connecting to Neo4j at %s as '%s'", uri, user)
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")


def get_driver() -> Driver:
    """Return the shared Neo4j Driver (connection pool).

    The driver is created on first call and reused for the lifetime of the
    process.  The underlying connection pool is managed by the neo4j-python
    driver itself.
    """
    return _Neo4jSingleton().get_driver()
