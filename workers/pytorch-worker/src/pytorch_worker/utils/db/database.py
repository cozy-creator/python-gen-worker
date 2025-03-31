import os
import logging
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

_db_connection = None

def get_db_connection():
    """
    Returns a connection to the database. Creates a new connection if one doesn't exist.
    Uses the DB_DSN environment variable for connection details.
    """
    global _db_connection
    
    if _db_connection is None:
        db_dsn = os.getenv("DB_DSN")
        if not db_dsn:
            raise ValueError("DB_DSN environment variable not set")
        
        try:
            logger.info("Connecting to database...")
            _db_connection = psycopg2.connect(db_dsn, cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    return _db_connection

def close_db_connection():
    """
    Closes the database connection if it exists.
    """
    global _db_connection
    
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed")