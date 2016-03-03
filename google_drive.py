'''
Download data from google drive using basic python packages.

MOSTLY BROKEN - DO NOT ATTEMPT TO USE AT THIS TIME
'''

# Darien Morrow - darienmorrow@gmail.com - dmorrow3@wisc.edu
# Blaise Thompson - blaise@untzag.com


### import ####################################################################


import os
import urllib


### helper methods ############################################################


def filenamegetter(fileID):
    """
    Return the filename corresponding to a public Google Drive ID.
    
    Attributes
    ----------
    fileID : string    
        Unique Google Drive ID that may correspond to a folder if desired.        
        
    Returns
    -------
    string
        A string that is the human readable filename of the inputed file ID.
        If applicable the file extension of the file is included in the filename.
    """    

    urltoopen = ''.join(['https://drive.google.com/file/d/',fileID,'/view?pref=2&pli=1'])    
    print urltoopen    
    gdrivecontent0 = urllib.urlopen(urltoopen).readlines()  
    title = gdrivecontent0[0].split("<title>")[1].split("- Google Drive")[0]    
    return title

def filedownloader(fileID, savelocation):
    """
    Download file corresponding to a public Google Drive ID.
    
    Attributes
    ----------
    fileID : string    
        Unique Google Drive ID that may NOT correspond to a folder currently.        
    save location : string
        Valid, local file path to a directory in which to save file.

    Returns
    -------
    string
        A string that is the human readable filename of the inputed file ID.
        If applicable the file extension of the file is included in the filename.

    Examples
    --------
    gdrive_filedownloader('0B8z-JGr_8g4RWm0wb0tDWjZUcWc','/Users/darienmorrow/Desktop/')
    """     
 
    urllib.urlretrieve('https://docs.google.com/uc?export=download&id='+fileID ,savelocation+gdrive_filenamegetter(fileID))


def contentelucidator(fileID, titleYN, folderYN, contentsYN):
    """
    Return information about file or folder corresponding to a public Google Drive ID.
     
    Attributes
    ----------
    fileID : string    
        Unique Google Drive ID that may correspond to a folder.        
    titleYN : Boolean
        Toggles if function prints retrieved filename of fileID.
    folderYN : Boolean
        Toggles if function will return statement that states if fileID 
        corresponds to a file or folder.
    contentsYN : Boolean 
        Toggles if function will return all info retrieved from fileID URL.      

    Returns
    -------
    string
        Title and file extension of fileID.
    string
        Statement concerning identity of fileID:        
        'is not folder' or 'is folder'
    string
        Many lines of strings that are all the info hijacked from Google.

    Examples
    --------
    An example of a folder:    
    gdrive_contentelucidator('0B5XBhboKMJCTbFR4WUY0TG5QajA', True, True, True)

    An example of a file:
    gdrive_contentelucidator('0B8z-JGr_8g4RWm0wb0tDWjZUcWc', True, True, True)  
    """
    
    # /d/ is download statement
    urltoopen = ''.join(['https://drive.google.com/file/d/',fileID,'/view?pref=2&pli=1'])    
    gdrivecontent0 = urllib.urlopen(urltoopen).readlines()  
    title = gdrivecontent0[0].split("<title>")[1].split("- Google Drive")[0]    
    
    if titleYN == True:
        print title
    if folderYN == True:     
        if 'image' in gdrivecontent0[0]:
            print 'is not folder' 
        else:
            print 'is folder'
    if contentsYN == True:
        for element in gdrivecontent0:
            print element


### main download method ######################################################


def download(fileID, output_folder=None):
    # TODO:
    urltoopen = ''.join(['https://drive.google.com/folderview?id=',fileID,'&usp=sharing'])    
    gdrivecontent0 = urllib.urlopen(urltoopen).readlines()  
    # Title corresponding to fileID.
    title = gdrivecontent0[0].split("<title>")[1].split("</title>")[0]    
    print title

    for element in gdrivecontent0:
        print 'new line______________'  
        print ''
        print element
    
    print 'new new line______________'
    frank = gdrivecontent0[2].split(',')
    print frank[-1]
    jill = str(frank[-1])
    print jill[15]    
    print jill[15] == "]"
    print jill == str(frank[-1])


### testing ###################################################################


if __name__ == '__main__':
    #folder
    #gdrive_universaldownloader('0B5XBhboKMJCTbFR4WUY0TG5QajA')
    #file
    download('0B8z-JGr_8g4RWm0wb0tDWjZUcWc')
    download('0B5XBhboKMJCTbFR4WUY0TG5QajA')  
    
    if False:
        contentelucidator('0B5XBhboKMJCTbFR4WUY0TG5QajA', True, True, True)
        print filenamegetter('0B5XBhboKMJCTbFR4WUY0TG5QajA')
        filedownloader('0B8z-JGr_8g4RWm0wb0tDWjZUcWc','/Users/darienmorrow/Desktop/')
