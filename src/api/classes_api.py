import os
import json
import logging
import traceback
from typing import Union

from src.logic import Param, Settings
from src.api import DataDBAPI


class ClassAPI:

    def __init__(self):
        self.classes = {}
        self.current_class_name = None
        self.is_opened = False
        self.is_saved = False

        self.settings = Settings()
        self.settings.path = 'settings_classes.json'

        self.settings.classes_local_path = Param(
            '',
            text={
                'en': 'local path to classes.json',
                'ru': 'путь к файлу classes.json'
            }
        )

        # self.settings.username = Param(
        #     val='',
        #     text={
        #         'en': 'Username',
        #         'ru': 'Имя пользователя'
        #     }
        # )
        # self.settings.password = Param(
        #     val='',
        #     text={
        #         'en': 'Password',
        #         'ru': 'Пароль'
        #     }
        # )
        # self.settings.hostname = Param(
        #     val='',
        #     text={
        #         'en': 'host of db api',
        #         'ru': 'хост апи базы данных'
        #     }
        # )
        # self.settings.db_name = Param(
        #     '',
        #     text={
        #         'en': 'database name',
        #         'ru': 'Имя базы данных'
        #     }
        # )
        self.data_db_api: DataDBAPI = None

    def open_local(self, classes_local_path: str = None) -> bool:
        """
        Opens classes dict from local path and set current_class_name to first class
        :return: True at success, False otherwise
        """
        if classes_local_path is not None:
            self.settings.classes_local_path.val = classes_local_path
        if len(self.settings.classes_local_path.val) > 0:
            if os.path.isfile(self.settings.classes_local_path.val):
                try:
                    with open(self.settings.classes_local_path.val, 'r') as f:
                        self.classes = json.load(f)
                        self.current_class_name = self.class_names()[0]
                    self.is_opened = True
                except Exception:
                    logging.error(f"{self.__module__}.{self.__class__.__name__}: open_local: \n {traceback.format_exc()}")
                    self.is_opened = False
            else:
                self.is_opened = False
        else:
            self.is_opened = False

        return self.is_opened

    def save_local(self, classes_local_path: str = None) -> bool:
        if classes_local_path is not None:
            self.settings.classes_local_path.val = classes_local_path
        if len(self.settings.classes_local_path.val) > 0:
            try:
                with open(self.settings.classes_local_path.val, 'w') as f:
                    json.dump(self.classes, f)
                self.is_saved = True
            except Exception:
                logging.error(f"{self.__module__}.{self.__class__.__name__}: save_local: \n {traceback.format_exc()}")
                self.is_saved = False
        else:
            self.is_saved = False
        return self.is_saved

    def save_to(self, classes=None):
        if isinstance(classes, ClassAPI):
            classes.classes = self.classes
            classes.current_class_name = self.current_class_name

    # def open_db(self, username: str = None, password: str = None, hostname: str = None, db_name: str = None) -> bool:
    def open_db(self, data_db_api: DataDBAPI, db_name: str):
        """
        Opens classes dict from db uri and set current_class_name to first class
        :return: True at success, False otherwise
        """

        self.data_db_api = data_db_api
        self.id_opened = False
        # if username is not None:
        #     self.settings.username.val = username
        # if password is not None:
        #     self.settings.password.val = password
        # if hostname is not None:
        #     self.settings.hostname.val = hostname
        # if db_name is not None:
        #     self.settings.db_name.val = db_name
        # self.data_db_api = DataDBAPI(username=username, password=password, hostname=hostname)
        if self.data_db_api.check() and (self.data_db_api.authorized or self.data_db_api.auth_login()):
            self.classes = self.data_db_api.image_data_get_classes(db_name=db_name)
            if self.__len__() > 0:
                self.current_class_name = self.class_names()[0]
                self.id_opened = True
            else:
                self.current_class_name = None
                self.id_opened = False
        else:
            self.classes = []
            self.current_class_name = None
            self.is_opened = False
        return self.is_opened

    # def save_db(self, username: str = None, password: str = None, hostname: str = None, db_name: str = None) -> bool:
    def save_db(self, data_db_api: DataDBAPI, db_name: str):
        # if username is not None:
        #     self.settings.username.val = username
        # if password is not None:
        #     self.settings.password.val = password
        # if hostname is not None:
        #     self.settings.hostname.val = hostname
        # if db_name is not None:
        #     self.settings.db_name.val = db_name
        # self.data_db_api = DataDBAPI(username=username, password=password, hostname=hostname)
        self.data_db_api = data_db_api
        if self.data_db_api.check() and (self.data_db_api.authorized or self.data_db_api.auth_login()):
            classes = self.data_db_api.image_data_set_classes(db_name=db_name, classes=self.classes)
            if self.__len__() == len(classes):
                self.is_saved = False
            else:
                self.is_saved = True
        else:
            self.is_saved = False
        return self.is_saved

    def class_names(self) -> list:
        """
        Returns class names as list
        :return: list of str
        """
        return [class_dict["class_name"] for class_dict in self.classes if "class_name" in class_dict]

    def __len__(self) -> int:
        """
        Returns number of classes
        :return:
        """
        return len(self.classes)

    def get(self) -> str:
        """
        Gets current class name
        :return: str - current class name
        or None if self.classes list is empty
        """
        return self.current_class_name

    def set(self, class_name: str) -> bool:
        """
        Tries to set current class name

        :param class_name:
        :return: True on success, False if class_name not in self.classes
        """
        if class_name in self.classes:
            self.current_class_name = class_name
            return True
        else:
            return False

    def mask_color(self, class_name: str = None) -> Union[str, None]:
        """

        If 'class_name' is not None
        Returns color of 'class_name'
        else returns color of current_class_name
        :param class_name:
        :return: list of ints like [128, 128, 128, 128] that are represent 'mask_color'
        or None if 'class_name' not in self.classes
        if 'class_name' is None, return color of current_class_name.
        If current_class_name is None, return None
        """
        if class_name is None:
            if self.current_class_name in self.classes:
                return self.classes[self.current_class_name]['mask_color'].copy()
            else:
                return None
        else:
            if class_name in self.classes:
                return self.classes[class_name]['mask_color'].copy()
            else:
                return None
