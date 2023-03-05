## NYT Book Crawler

``weekly_update.py`` is a python script that makes recursive API call in between the last and the latest weekly release.

The after storing them in a dataframe format it appends them to the SQL database
 


For triggering the script every monday of the week, run the following command:

0 3 * 1 * python weekly_update.py

| | | | |<br>
| | | | ----- Day of week (0 - 7) (Sunday=0 or 7)<br>
| | | ------- Month (1 - 12)<br>
| | --------- Day of month (1 - 31)<br>
| ----------- Hour (0 - 23)<br>
------------- Minute (0 - 59)<br>


the last week parsed is stored in the file ``last_week``