# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:48:58 2021

@author: yebin
"""

import analysis
import crawling
import database

__all__ = ['analysis', 'crawling', 'database']


# 절대 참조 예시
'''
from roboadvisor import analysis
from roboadvisor import crawling
from roboadvisor import database
'''

# 상대 참조 예시
'''
from .series import series_test
from .crawling.parser import parser_test
'''
