from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Lot(db.Model):
    id = db.Column(db.Integer, primary_key=True) 
    location = db.Column(db.String(64), index=True)
    total_spots = db.Column(db.Integer, index=True)
    available_spots = db.Column(db.Integer, index=True)
    percentage = db.Column(db.String(64), index=True)
    
    spots = db.relationship('Spot', backref='Location', lazy='dynamic')

    def __repr__(self):
        return '{}'.format(self.location) 

class Spot(db.Model): 
    id = db.Column(db.Integer, primary_key=True)  
    availability = db.Column(db.String(64), index=True)
    lot_location = db.Column(db.Integer, db.ForeignKey('lot.id'))

    def __repr__(self):
        return '<Spot {}>'.format(self.id)


