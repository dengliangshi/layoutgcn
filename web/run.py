#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries
from flask_script import Manager, Shell
from flask_migrate import Migrate, MigrateCommand, upgrade

# User define module
from app import create_app, db
from app.models import Sample, Label

# ------------------------------------------------------Global Variables----------------------------------------------------
app = create_app()

manager = Manager(app)
migrate = Migrate(app, db)


# -----------------------------------------------------------Main-----------------------------------------------------------
def make_shell_context():
    return dict(app=app, db=db, Sample=Sample, Label=Label)
manager.add_command("shell", Shell(make_context=make_shell_context))
manager.add_command('db', MigrateCommand)


@manager.command
def deploy():
    db.create_all()
    upgrade()


if __name__ == '__main__':
    manager.run()
