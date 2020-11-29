import threading
import requests
from typing import Dict, List
from src.utils.utils import formaturl


class DataDBAPI:
    """
    Module for Data DB API

    """

    def __init__(
            self,
            host: str = 'localhost',
            username: str = None,
            password: str = None,
            access_token: str = None,
            update_token_in_thread: bool = True,
            token_expire_time: int = 3600
    ):
        self.host = formaturl(host)
        self.timeout = 3
        self.access_token = access_token
        self.username = username
        self.password = password
        self.update_token_in_thread = update_token_in_thread
        self.token_expire_time = token_expire_time
        self.authorized = False

    def check(self) -> bool:
        link = f"{self.host}/docs"
        try:
            response = requests.get(
                url=link,
                timeout=self.timeout
            )
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            return False

    def update_token_thread(self):
        """
        try update token, then sleep for a `self.token_expire_time`
        if unsuccessfully, stop thread
        """
        if not self.auth_refresh():
            if not self.auth_login(self.username, self.password):
                return

        threading.Timer(
            max(self.token_expire_time - 120, 1),
            self.update_token_thread
        ).start()

    def users_get_me(self) -> Dict:
        link = f"{self.host}/users/me"
        try:
            response = requests.get(
                url=link,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                return params
            else:
                return {}
        except Exception as e:
            return {}

    def auth_login(self, username: str, password: str) -> bool:
        self.username = username
        self.password = password

        link = f"{self.host}/auth/jwt/login"
        data = {
            "username": self.username,
            "password": self.password
        }
        try:
            response = requests.post(
                url=link,
                data=data,
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                self.access_token = params.get('access_token', None)
                if self.update_token_in_thread:
                    self.update_token_thread()
                self.authorized = True
                return True
            else:
                return False
        except Exception as e:
            return False

    def auth_refresh(self) -> bool:
        link = f"{self.host}/auth/jwt/refresh"

        try:
            response = requests.post(
                url=link,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                self.access_token = params.get('access_token', None)
                self.authorized = True
                return True
            else:
                print(response.status_code, response.content)
                return False
        except Exception as e:
            return False

    def image_data_get_classes(self, db_name: str):
        pass

    def image_data_set_classes(self, db_name: str, classes: dict):
        pass

    def image_data_get_image_data(self, db_name: str, task_name: str, id: str = None):
        pass

    def image_data_post_image_data(self, db_name: str, task_name: str, image_data: dict, id: str = None):
        pass

    def image_data_delete_image_data(self, db_name: str, task_name: str, id: str):
        pass

    def image_data_get_tasks(self) -> List[Dict]:
        """get tasks"""
        assert self.authorized, "Unauthorized"
        link = f"{self.host}/tasks"

        try:
            response = requests.get(
                url=link,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                return params.get('data', [])
            else:
                print(response.status_code, response.content)
                return []
        except Exception as e:
            return []

    def image_data_post_tasks(self, db_name: str, task_name: str) -> bool:
        """create new task"""
        assert self.authorized, "Unauthorized"
        link = f"{self.host}/tasks"

        try:
            params = {
                "db_name": db_name,
                "task_name": task_name
            }
            response = requests.post(
                url=link,
                params=params,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                return params.get('data', False)
            else:
                print(response.status_code, response.content)
                return False
        except Exception as e:
            return False

    def image_data_delete_tasks(self, db_name: str, task_name: str) -> bool:
        assert self.authorized, "Unauthorized"
        link = f"{self.host}/tasks"

        try:
            params = {
                "db_name": db_name,
                "task_name": task_name
            }
            response = requests.delete(
                url=link,
                params=params,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                return params.get('data', False)
            else:
                print(response.status_code, response.content)
                return False
        except Exception as e:
            return False

    def image_data_add_dbs(self, db_name: str) -> bool:
        assert self.authorized, "Unauthorized"
        link = f"{self.host}/dbs"

        try:
            params = {
                "db_name": db_name
            }
            response = requests.post(
                url=link,
                params=params,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                return params.get('data', False)
            else:
                print(response.status_code, response.content)
                return False
        except Exception as e:
            return False

    def image_data_delete_dbs(self, db_name: str) -> bool:
        assert self.authorized, "Unauthorized"
        link = f"{self.host}/dbs"

        try:
            params = {
                "db_name": db_name
            }
            response = requests.delete(
                url=link,
                params=params,
                headers={'Authorization': f"Bearer {self.access_token}"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                params = response.json()
                return params.get('data', False)
            else:
                print(response.status_code, response.content)
                return False
        except Exception as e:
            return False
