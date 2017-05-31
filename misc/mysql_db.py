import _mysql


def get_db(host, user, port):
    """ This function connects to the specified mysql database.

    Args:
        host, user, port (strings): the appropriate mysql connection information

    Returns:
        MySQL database connection
    """

    db = _mysql.connect(host=host, user=user, port=port)
    db.set_character_set('utf8')
    db.query('SET NAMES utf8;')
    db.query('SET CHARACTER SET utf8;')
    db.query('SET character_set_connection=utf8;')
    return db

def execute_query(db, query, how=1):
    """ This function executes the specified query from the given database.
    It returns the results as a tuple of the rows. By default, each element
    of the tuple is a dictionary mapping from the column name to the value
    for that row.

    N.B. The entire result is stored in memory, so this is not efficient
    for queries which return large, mostly useless, result sets.

    Args:
        db (MySQL database connection): the database connection object, 
            presumably created with get_db

        query (string): the sql query. No safety checking is performed.

        how (int): the format of the return rows. See http://mysql-python.sourceforge.net/MySQLdb.html
            for more details.

    Returns:
        tuple: a list (tuple) of all of the rows matching the query
    """

    db.query(query)
    r = db.store_result()
    rows = r.fetch_row(maxrows=0, how=how)
    return rows


def list_databases(db):
    """ This function returns a list of all of the databases associated
    with the MySQL connection.

    Args:
        db (MySQL database connection): the database connection object, 
            presumably created with get_db

    Returns:
        list: a list of all of the databases.

    """

    query = "show databases;"
    how = 0 # return things as tuples
    res = execute_query(db, query, how=0)
    databases = []

    for r in res:
        s = r[0]
        databases.append(s.decode('ascii'))
    
    return databases

def list_tables(db, database=None):
    """ This function returns a list of all of the tables in the given
    database of the MySQL connection.

    Args:
        db (MySQL database connection): the database connection object, 
            presumably created with get_db

        database (string): the name of the database to use. If the database
            is not specified, the function assumend "select_db" has already
            been called on the db object.

    Returns:
        tuple: a list of the tables.

    Stateful change:
        db: the selected database for db will be <database>
    """

    query = "show tables;"
    how = 0 # return things as tuples
    res = execute_query(db, query, how=how)
    if database is not None:
        db.select_db(database)

    tables = []

    for r in res:
        s = r[0]
        tables.append(s.decode('ascii'))

    return tables
