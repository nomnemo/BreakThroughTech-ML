#!/usr/bin/env python
# coding: utf-8

# # Using Command Line Tools To View Data

# ##  Should I Open a Data File via a File Management Browser? Think Again 

# You already encountered ways to quickly overview what your data looks like using Pandas. (Don't worry, if memory fails, we will revisit that material below!)<br>
# 
# It is often useful to get the initial impression of the data before loading it into a Pandas DataFrame. <br>
# How does one look at file contents? We are all used to visual interfaces, like the file browser on your computer, and it may be tempting to simply navigate (using a file explorer on WIndows or Finder on Mac) to where your data file is located and open it. However, there are a few good reasons not to do so:<br>
# - The file may be in a **format** that is not immediately readable by a text editor (one example of such formats is `.json`, and it is very commonly used)
# - The file may be **prohibitively large**: it may contain so many lines that trying to open it in a text editor may hang your software.
# - For most purposes, you do not strictly need to look at *all* entries &mdash; usually, what we are after is simply getting a sense of how many rows and columns there are, how they are separated, and what is the type of values stored in each column. Hence, we **don't need to view all lines**, and trying to load the whole file would be overkill.
# - You may be working off **terminal / command line**, where you may not have access to window-based file browsing nor a text editor software with a nice GUI (graphical user interface).
# 
# Let's talk about this last one. In machine learning jobs, you will often execute your code on a *remote server* in the cloud. That is, you will only use *your* machine to communicate with a (powerful) *remote* machine, and all the computation will be running not on your computer, but on that remote server. You will only use your machine to either edit and upload your code to a cloud instance, or you will edit it remotely and your machine will be a mere access interface to edit, manipulate, and run your project in the cloud. While working on a cloud server instance, it is typical to channel commands via the **command line**, where the options for displaying file contents are limited, and the end result is not very easy to read. 
# 
# <br>
# For all of these reasons, it is best to avoid opening data files using a command line text editing software, and instead develop a habit of doing the initial data exploration via  command line commands (even if you are working in Jupyter Notebooks!). Below, we show how to do this.
# 
# Note: When talking about channeling **Linux** commands via the **command line**, people will sometimes use terms such as **terminal** or **console** instead. Strictly, the terminal and the console are *interfaces* that allow you to access the command line. You can use any of these terms to refer to sending commands to your computer using a text interface, and people will generally know what you mean.
# <br>

# ## Initial Data Exploration with the *Command Line* Tools
# 
# You will work with our usual `censusData` dataset located in the folder named `data`.
# 
# ### Get the Number of Lines: `cat` --> `wc -l`
# 
# Examine and run the command below. 

# In[ ]:


get_ipython().system(' cat data/censusData.csv | wc -l')


# The first part of the line above is the `cat` command. What `cat` does is it prints the content of a file (generally `cat` can also be used to write text into a file).
# 
# 
# Next is the input that we pass to the `cat` function: the file path and name.<br>
# 
# After that comes the *pipe command*: the vertical bar symbol `|`. It lets you chain two commands such that the output of the first is the input to the second.<br>
# 
# The second command in question is the `wc` command, for which we specified the option `-l`. While `wc` stands for "word count", once we pass the option `-l` to it, we specify that we want to get a count of the *lines*.

# ### Print the First Few Lines of the Data File: `head`

# Note that the total count that we got from piping `cat` into `wc` is 7001. If you recall, our data contained 7000 lines of data entries. Hence, this count includes the very first line that contains just the column names. 
# 
# In general, you won't be able to benefit from such prior knowledge, and it's always a good idea to look at the first few lines just to see if the headers are included or not (some datasets specify column names in a separate file, while the data file itself only lists the values).
# 
# Let's print the first few lines using command line tools:

# In[ ]:


get_ipython().system(' head -5 data/censusData.csv')


# Doing this, you can instantly see if the column names are included in the data or not. You see that they are. Therefore, the true number of data entries is 7000.

# ### Get the Number of Columns: `head` --> `tr` --> `wc`

# You can get the number of columns using the command line in this manner:
# - get the first line containing column names
# - split the first line into multiple lines, in which each column name will be on its own line
# - count the lines
# 
# We already know how to do the first and the last steps. The middle step is taken care of by the `tr` command:

# In[ ]:


get_ipython().system(" head -1 data/censusData.csv | tr ',' '\\n' | wc -l")


# In[ ]:


get_ipython().system(" head -1 data/censusData.csv | tr ',' '\\n'")


# The *translate* command `tr` takes in two parameters: the original symbol or string to replace, and what to replace it with. Here, we are looking for the commas, and "translating" them into new lines. 
# 
# We see that the number of columns is 15.
# 
# Note: If we printed the first few lines of a data file using `head` and saw that our data columns are separated not with a comma but, say, with a tab (which is more typical than a comma delimiter), we would use `tr '\t' '\n'` instead.
# 

# ## Initial Data Exploration with Pandas

# The cells below are a reminder of how we could answer all of the same questions about our data in Python:

# In[ ]:


import pandas as pd
import os 


# In[ ]:


filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)


# In[ ]:


df.head()


# In[ ]:


df.shape


# You can see that the dimensions we got using command line tools are the same as the ones put out by Pandas tools.
