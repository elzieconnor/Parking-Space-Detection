from flask import Flask, Response, request, redirect, render_template, session, url_for
from flask_admin import Admin, BaseView, expose
from flask_sqlalchemy import SQLAlchemy
from flask_admin.contrib.sqla import ModelView
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from apscheduler.schedulers.background import BackgroundScheduler

from lib.img_proc import ImgProcessor
from storage.database.models import *
from lib.cam import Camera
from os import listdir
from os.path import isfile, join

import storage.database.models
import numpy as np
import requests
import atexit
import datetime
import json
import yaml
import cv2
import os, os.path

#   Global Variables [TODO: REMOVE IF POSSIBLE]

id_increment = 1
object_id = None
timestamp = datetime.datetime.now().time() 
location = None

#   Flask server initialization
#   Flask configuration handling

application = Flask(__name__)

application.config.from_object(__name__)
application.config.update(dict(
	SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(application.root_path + '/storage/database', 'database.db'),
	SQLALCHEMY_TRACK_MODIFICATIONS = False,
	SECRET_KEY = 'UHu9qUw9SLgKvneuJXmsQRfV',
	JSONIFY_MIMETYPE = 'application/json',
	APPLICATION_ROOT = '/',
	TESTING = False,
	DEBUG = True,
))

#   Database Configuration

db.init_app(application)
application.app_context().push()
db.create_all()

''' ------------------------------------------------------------------------
	Administrator Portal
	
	Configuration form view takes input for camera_id [to retrieve camera object]
	and Lot record. Additional views allow admin to manipulate database,
	including create new records, edit records, and delete records.
	------------------------------------------------------------------------ ''' 

#   Lot Configuration Flask Form

class ConfigurationForm(FlaskForm):
	camera_id = StringField(
		'Video Stream ID',
		validators = [DataRequired()])
	
	lot_information = QuerySelectField(
		'Video Stream Location',
		query_factory = lambda: Lot.query,
		allow_blank = False)
	
	submit = SubmitField('Submit')

#   Admin Portal Views [TODO: HTML/CSS]

class ConfigurationView(BaseView):
	@expose(url = '/', methods = ('GET', 'POST'))
	def index(self):
		global object_id, location
		form = ConfigurationForm()
		if form.validate_on_submit():
			object_id = form.camera_id.data
			location = form.lot_information.data
			return redirect('/admin/configuration/display')
		return self.render('configuration.html', form = form)

admin = Admin(application, name = 'Admin Portal', template_mode = 'bootstrap3')

admin.add_view(ConfigurationView(name = 'Configuration', endpoint = 'configuration'))
admin.add_view(ModelView(Lot, db.session, category = "Database"))
admin.add_view(ModelView(Spot, db.session, category = "Database"))

''' ------------------------------------------------------------------------
	Vehicle Detection Process:
	
	APScheduler object runs update_availability function every 30 seconds.
	System verifies that camera object and configuration file exists. If so,
	ImgProcessor() class detects availability and returns results as a list.
	The list and Lot ID are stored in database to be displayed on GUI.
	------------------------------------------------------------------------ '''   

#   Database insert function

def database_import(avail_list, lot_id):
	# Calculate total spots, total available spots, and percentage of available spots
	spot_num = len(avail_list)
	avail_num = avail_list.count('Available')
	unavail_num = avail_list.count('Unavailable')
	percent_spots = str(int((unavail_num / spot_num)*100)) + '%'
	
	with application.app_context():
		# Update Lot table totals based on availability information
		update = Lot.query.get(lot_id)
		update.total_spots = spot_num
		update.available_spots = avail_num
		update.percentage = percent_spots
		
		# Delete previous availability data based on Lot ID
		db.session.query(Spot).filter(Spot.lot_location == lot_id).delete()
		for x in avail_list:
			# Insert availability data and Lot ID into Spot table
			imported = Spot(availability = x, lot_location = lot_id)
			db.session.add(imported)
		db.session.commit()
	
#   Scheduler Configuration

def update_availability():
	# Global variables [TODO: REMOVE IF POSSIBLE]
	global timestamp
	
	# Update scheduler timestamp
	timestamp = datetime.datetime.now().time()
	#detection_process = ImgProcessor()
	#availability_information = detection_process.process_frame()
	#availability_information = detection_process.process_frame()
	#database_import(availability_information, 1)
	configuration_path = application.root_path + "/storage/config"
	config_filenames = [f for f in listdir(configuration_path) if isfile(join(configuration_path, f))]
	if os.listdir(application.root_path + "/storage/config"):
		file_num = len([f for f in os.listdir(configuration_path)if os.path.isfile(os.path.join(configuration_path, f))])
		for x in config_filenames:
			# Retrive frame from video stream via ID
			#camera_id = x[:-4]
			#r = requests.get('http://127.0.0.1:8080/get_frame', headers = {'cam_id': camera_id})
			#data = r.content
			#frame = json.loads(data.decode("utf8"))
			#frame = np.asarray(frame, np.uint8)
			#frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
			
			# Process frame and return availability information as list
			detection_process = ImgProcessor()
			availability_information = detection_process.process_frame(x)
			database_import(availability_information[0], availability_information[1])
	else:
		pass

scheduler = BackgroundScheduler()
scheduler.add_job(func = update_availability, trigger = "interval", seconds = 10)
scheduler.start()
# Shutdown scheduler object when server is shutdown
atexit.register(lambda: scheduler.shutdown())

''' ------------------------------------------------------------------------
	Server Routes:
	
	Index will display contents of Lot tables, including total number of
	spots, total number of available spots, and the percentage of available
	spots. Info will display spot availability for each respective lot
	using URL parameters. Get_frame returns a frame based on camera_id.
	------------------------------------------------------------------------ ''' 

#   index Route

@application.route('/')
@application.route('/index')
def index():
	# Query Lot table and store records into data variable
	# Push data variable to HTML template
	data = db.session.execute("SELECT * FROM Lot").fetchall()
	return render_template('index.html', data = data, timestamp = timestamp)

#   info Route
#   Uses 'location' URL parameter to determine records to fetch

@application.route('/info/<location>')
def info(location):
	# Query Spot table and store records into data variable
	# Push data variable to HTML template
	data = db.session.execute("SELECT * FROM Spot WHERE lot_location = :lot_location;", {"lot_location": location}).fetchall()
	return render_template('info.html', data = data, timestamp = timestamp)

#   get_frame Route

@application.route('/get_frame', methods=['GET'])
def get_frame():
	# Retrieve frame object via object_id variable
	# Encode frame and return it
	r = requests.get('http://10.13.78.214:8080/get_frame', headers = {"cam_id": object_id}).content
	frame = json.loads(r)
	frame = np.asarray(frame, np.uint8)
	frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
	r, jpg = cv2.imencode('.jpg', frame)
	return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

#   Lot Configuration Route

@application.route('/admin/configuration/display', methods = ['GET', 'POST'])
def display():
	# Global variables [TODO: REMOVE IF POSSIBLE]
	global location, id_increment

	#id_increment = 1
	config_information = {'id': 1, 'lot': 0, 'points': []}
	arr = []
	# Get data from Javascript function
	# If data doesn't equal None [boundary box has been drawn],
	# Create YAML file named after object_id variable
	# Dump box coordinates into YAML file
	data = request.json
	if data != None:
		config_information['points'] = [list(data[0]), list(data[1]), list(data[2]), list(data[3])]
		config_information['id'] = id_increment
		config_information['lot'] = location.id
		id_increment = id_increment + 1
		arr.append(config_information)
		with open('./storage/config/' + object_id + '.yml','a') as yamlfile:
			yaml.dump(arr, yamlfile)
	return render_template("test.html")

#   Run server on 'localhost:8090'

if __name__ == '__main__':
	application.run(host = '0.0.0.0', port = '8090', debug = False)
