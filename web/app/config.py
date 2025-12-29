#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------
base_dir = os.path.abspath(os.path.dirname(__file__))

# -----------------------------------------------------------Main-----------------------------------------------------------
class Config(object):

    SSL_DISABLE = True

    WTF_CSRF_ENABLED = False

    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True

    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(base_dir, "database/data.sqlite")
    

    @staticmethod
    def init_app(app):
        pass
