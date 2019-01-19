import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def select_all_estimators(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_Estimator")
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)

def select_all_estimator_parameters(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_EstimatorParameter")
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)

def select_all_transformers(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_Transformer")
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)

def select_all_transformer_parameters(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_TransformerParameter")
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)

def select_all_feature_selectors(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_NonEstimatorFeatureSelector")
 
    rows = cur.fetchall()
 
    for row in rows:
        print(row)

def select_all_feature_selectors_parameters(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM T_NonEstimatorFeatureSelectorParameter")
 
    rows = cur.fetchall()
 

    for row in rows:
        print(row)

    print("***** AND *****")
    
    cur.execute("SELECT * FROM T_Estimator a WHERE a.F_Estimator_CanFeatureSelect=1")
 
    rows = cur.fetchall()
 

    for row in rows:
        print(row)

def main():
    database = "streamml.db"
 
    # create a database connection
    conn = create_connection(database)
    
    print("1. Query all Estimators")
    select_all_estimators(conn)

    print("2. Query all Parameters")
    select_all_estimator_parameters(conn)

    print("3. Query all Transformers")
    select_all_transformers(conn)

    print("4. Query all Transformer Parameters")
    select_all_transformer_parameters(conn)
 
    print("5. Query all NonEstimatorFeatureSelectors")
    select_all_feature_selectors(conn)

    print("6. Query all FeatureSelector Parameters")
    select_all_feature_selectors_parameters(conn)
 
if __name__ == '__main__':
    main()
