from pathlib import Path
import threading
import requests
import numpy as np
import logging
from typing import Dict, List, Union
from src.utils.utils import formaturl, image_to_stream


class DataStoreAPI:
    """
    Module for Data Store API

    """

    def __init__(
            self,
            hostname: str = 'localhost',
            username: str = None,
            password: str = None,
            access_token: str = None,
            update_token_in_thread: bool = True,
            token_expire_time: int = 3600
    ):
        self.hostname = formaturl(hostname)
        self.timeout = 3
        self.access_token = access_token
        self.username = username
        self.password = password
        self.update_token_in_thread = update_token_in_thread
        self.token_expire_time = token_expire_time
        self.authorized = False

    def check(self) -> bool:
        link = f"{self.hostname}/docs"
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
            logging.error(f"{self.__module__}.{self.__class__.__name__}.check(): \n {e}")
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
        link = f"{self.hostname}/users/me"
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
            logging.error(f"{self.__module__}.{self.__class__.__name__}.users_get_me(): \n {e}")
            return {}

    def auth_login(self, username: str, password: str) -> bool:
        self.username = username if username is not None else self.username
        self.password = password if password is not None else self.password

        link = f"{self.hostname}/auth/jwt/login"
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
            logging.error(f"{self.__module__}.{self.__class__.__name__}.auth_login(): \n {e}")
            return False

    def auth_refresh(self) -> bool:
        link = f"{self.hostname}/auth/jwt/refresh"

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
            logging.error(f"{self.__module__}.{self.__class__.__name__}.auth_refresh(): \n {e}")
            return False

    def store_post_array(self, image: np.ndarray) -> Union[List[str], None]:
        assert self.authorized, "Unauthorized"
        link = f"{self.hostname}/store"

        try:
            files = {"files": image_to_stream(image=image)}
            response = requests.post(
                url=link,
                files=files,
                headers={
                    'Authorization': f"Bearer {self.access_token}"
                },
                timeout=self.timeout * 10
            )
            if response.status_code == 200:
                return response.json().get('data', None)
            else:
                print(response.status_code, response.content)
                return None
        except Exception as e:
            logging.error(f"{self.__module__}.{self.__class__.__name__}.store_post_array(): \n {e}")
            return None

    def store_post_file(self, path: str) -> Union[List[str], None]:
        assert self.authorized, "Unauthorized"
        link = f"{self.hostname}/store"
        try:

            path = Path(path)
            if not path.is_file():
                return None
            files = {"files": open(path, 'rb')}

            response = requests.post(
                url=link,
                files=files,
                headers={
                    'Authorization': f"Bearer {self.access_token}",
                },
                timeout=self.timeout * 10
            )

            if response.status_code == 200:
                return response.json().get('data', None)
            else:
                print(response.status_code, response.content)
                return None
        except Exception as e:
            logging.error(f"{self.__module__}.{self.__class__.__name__}.store_post_file(): \n {e}")
            return None

    def store_delete(self, url: str) -> bool:
        assert self.authorized, "Unauthorized"
        link = f"{self.hostname}/store"

        try:
            response = requests.delete(
                link,
                json=[url],
                headers={
                    'Authorization': f"Bearer {self.access_token}"
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get('data', [False])[0]
            else:
                print(response.status_code, response.content)
                return False
        except Exception as e:
            logging.error(f"{self.__module__}.{self.__class__.__name__}.store_delete(): \n {e}")
            return False
