#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries


# User define module
from app import db

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Sample(db.Model):

    __tablename__ = "sample"

    id = db.Column(db.String(64), primary_key=True)
    image = db.Column(db.Text)
    blocks = db.Column(db.Text)
    predict_blocks = db.Column(db.Text)
    height = db.Column(db.Integer)
    width = db.Column(db.Integer)
    result = db.Column(db.Text)
    predict_result = db.Column(db.Text)

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def __repr__(self):
        return "<Sample %r>" % self.id
