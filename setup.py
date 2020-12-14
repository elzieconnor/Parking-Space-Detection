from setuptools import find_packages, setup

setup(
    name='flaskr',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask', 'flask-admin', 'flask-wtf', 'flask-sqlalchemy', 'opencv-python', 'matplotlib', 'pyyaml', 'requests', 'apscheduler'
    ],
)

