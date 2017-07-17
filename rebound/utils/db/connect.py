#!/usr/bin/python

import psycopg2
import sqlalchemy
#from config import config

def connect():
	"""Connect to the PostgreSQL database server"""
	conn = None
	try:
		# read connection parameters
		# params = config()

		# connect to the PostgreSQL server
		print('Connecting to the PostgreSQL database...')
		conn = psycopg2.connect("dbname=uo_rebound")

		# create a cursor
		cur = conn.cursor()

		# execute a statement
		print('PostgreSQL database version:')
		cur.execute('SELECT version()')
		
		# bind the connection to metadata 
		meta = sqlalchemy.MetaData(bind = conn) 

		# display the PostgreSQL database server version
		db_version = cur.fetchone()
		print(db_version)
		
		# close the communication with the PostgreSQL
		# cur.close()

		return conn, meta

	except(Exception, psycopg2.DatabaseError) as error:
		print(error)
	
	# finally:
		# if conn is not None:
			# conn.close()
			# print('Database connection closed.')

if __name__ == '__main__':
	connect()
