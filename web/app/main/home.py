#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries
from flask import Response, stream_with_context
from flask import render_template, request, redirect, url_for, flash

# User define module
from app import db
from app.main import main


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
@main.app_errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@main.route("/")
def index():
    return render_template("index.html")
