#!/usr/bin/python
# coding: utf-8

""" A simple implementation of the singleton design pattern. """

from threading import Lock

class SingletonMeta(type):
    """ This class is responsible for ensuring Singleton objects for the class it creates.
        Based on a thread-safe implementation of the Singleton design pattern by Refactoring Guru.
        https://refactoring.guru/design-patterns/singleton/python/example
    """

    # Contains the only instance of each class
    _instances = {}

    # Lock object is used to synchronize threads during first access to the Singleton.
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """ Creating only one instance for the class 'cls' """

        # Return only instance of class cls if the instance was already created
        if cls in cls._instances:
            return cls._instances[cls]

        # Else, create the instance with thread safe mechanism
        with cls._lock:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

            return cls._instances[cls]
