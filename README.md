### 1.Synopsis

A simple script to scrape Tweets using the Python package requests to retrieve the content and save it in database.

### 2.Installation and Requirements 
Streamlit : Used to create web apps

To install streamlit

  pip install streamlit
  
    import streamlit as st

PIL : Python Imaging Library (expansion of PIL) is the de facto image processing package for Python language. 
It incorporates lightweight image processing tools that aids in editing, creating and saving images.

To install pillow

  pip install pillow
  
    from PIL import Image
    
Snscrape: snscrape is a scraper for social networking services (SNS). 
It scrapes things like user profiles, hashtags, or searches and returns the discovered items, e.g. the relevant posts.

To install snscrape

  pip install snscrape
  
    import snscrape.modules.twitter as sntwitter
    

    import pandas as pd

    from pymongo import MongoClient
  
  

### 3.Output
      
          Datetime	        Tweet Id	      Text	      Username	        Reply Count	      Retweet Count	        Language	        Source	    Like Count



         2022-11-05 03:37:31+00:00	1.58873724471387E+018	@imVkohli Happy Birthday King Kohli..🎂🎊🎉	NileshKanake1	0	0	en	<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>	0
         
        

