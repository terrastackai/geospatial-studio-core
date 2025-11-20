# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import sys
import uuid
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from gfmstudio.auth.api_key_utils import apikey_expiry_date, generate_apikey

firstname = input("Enter first name:")
lastname = input("Enter last name:")
email = input("Enter email:")
user_id = str(uuid.uuid4())
created_at = updated_at = datetime.now().isoformat()


def create_user(db_url):
    """
    create user and api key in one transaction
    """
    key_id = str(uuid.uuid4())
    key_val = generate_apikey()
    hashed_key = key_val["hashed_key"]
    encrypted_key = key_val["encrypted_key"]
    expiry_date = apikey_expiry_date()

    sql_transaction = f"""
    BEGIN;

    -- create user
    INSERT INTO public."user"
    (first_name, last_name, email, data_usage_consent, organization_id, extra_data, id, created_at, updated_at, created_by, updated_by, active, deleted)
    VALUES('{firstname}', '{lastname}', '{email}', false, NULL, 'null'::json, '{user_id}'::uuid, '{created_at}', '{updated_at}', '{email}', '{email}', true, false);

    -- Then create the API key
    INSERT INTO public.apikey (value,hashed_key,last_used_at,user_id,expires_on,id,created_at,updated_at,created_by,updated_by,active,deleted) VALUES
	('{encrypted_key}','{hashed_key}',NULL,'{user_id}'::uuid,'{expiry_date}','{key_id}'::uuid,'{created_at}','{updated_at}','{email}','{email}',true,false);

    COMMIT;
    """
    try:

        subprocess.run(
            ["psql", "-d", db_url, "-c", sql_transaction], capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(e)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    db_url = os.getenv("AUTH_DATABASE_URI")
    if db_url:
        db_url = db_url.replace("+pg8000", "")
    # rprint(db_url)
    create_user(db_url=db_url)
