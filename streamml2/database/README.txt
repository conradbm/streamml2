**************************************
*** streamml database construction ***
**************************************
In order to set up our database we simply run `sh setup_db.sh`. This executes a series of python scripts using `sqlalchemy` the well known `ORM` framework that enables us to intuitively create tables with their relationships. I chose to construct a `sqlite` file database because of the ease of use the relatively small amount of data in the `sklearn` models and parameters leveraged. The main script that does the heavily lifting is the `database_engine_setup.py` which actually does web-scraping on several of the prebuilt sklearn models in the `streamml` ecosystem, drawing only the relevant parameters and descriptions needed to build convenient drop down menus and selections from a simple `distinct` query on a given model(s) the user has selected. 

TODO
Pre-processing tables, relationships, and builds.

TODO
Feature-selection tables, relationships, and builds.
Ensembling for several feature selectors.


