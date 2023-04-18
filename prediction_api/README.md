Local setup:
1. `cd prediction_api`
2. `python3 -m venv myenv`
3. `source myenv/bin/activate`
4. `pip install -r requirements.txt`
5. `python handler.test`

Deploy:
1. Install & run docker
2. `npm i`
3. `sls deploy --verbose`