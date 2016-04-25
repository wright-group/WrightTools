'''
Interact with google drive using the pydrive package.
'''

# Darien Morrow - darienmorrow@gmail.com - dmorrow3@wisc.edu
# Blaise Thompson - blaise@untzag.com


### import ####################################################################


import os
import time
import datetime


### define ####################################################################


directory = os.path.dirname(os.path.abspath(__file__))


### ensure google drive creds folder populated ################################


creds_dir = os.path.join(directory, 'temp', 'google drive')
if not os.path.isdir(creds_dir):
    os.mkdir(creds_dir)

mycreds_path = os.path.join(creds_dir, 'mycreds.txt')
if not os.path.isfile(mycreds_path):
    open(mycreds_path, 'a').close()


### drive class ###############################################################


class Drive:
    
    def __init__(self):
        # import pydrive
        import pydrive
        # authenticate
        self.mycreds_path = mycreds_path
        self._authenticate()

    def _authenticate(self):
        """
        This function, once run, will open up a login window in a web browser.
        The user must then athenticate via email and password to authorize the
        API for usage with that particular account. Note that 'mycreds.txt' may
        just be an empty text file. This function will create the correct
        dictionary structure in the file upon completion.
        """
        # This function requires a Client_secrets.json file to be in the
        # working directory.
        old_cwd = os.getcwd()
        os.chdir(directory)
        # import
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from pydrive.files import GoogleDriveFile
        # load
        self.gauth = GoogleAuth()
        self.gauth.LoadCredentialsFile(self.mycreds_path)
        if self.gauth.credentials is None:
            # authenticate if credentials are not found
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # refresh credentials if they are expired
            self.gauth.Refresh()
        else:
            # initialize the saved credentials
            self.gauth.Authorize()
        # finish
        self.gauth.SaveCredentialsFile(self.mycreds_path)
        self.api = GoogleDrive(self.gauth)
        os.chdir(old_cwd)
        
    def _list_folder(self, folder_id):
        # adapted from https://github.com/googledrive/PyDrive/issues/37
        # folder_id: GoogleDriveFile['id']
        _q = {'q': "'{}' in parents and trashed=false".format(folder_id)}
        raw_sub_contents = self.api.ListFile(_q).GetList()
        return [i['id'] for i in raw_sub_contents]

    def download(self, fileid, directory='cwd', overwrite=False, verbose=True):
        '''
        Recursively download from Google Drive into a local directory. By
        default, will not re-download if file passes following checks:
        1. same size as remote file
        2. local file modified after remote file
        
        Parameters
        ----------
        fileid : str
            Google drive id for file or folder.
        directory : str (optional)
            Local directory to save content into. By default saves to cwd.
        overwrite : bool (optional)
            Toggle forcing file overwrites. Default is False.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        
        Returns
        -------
        pydrive.files.GoogleDriveFile
        '''
        # get directory
        if directory == 'cwd':
            directory = os.getcwd()
        # get file object
        f = self.api.CreateFile({'id': fileid})
        f_path = os.path.join(directory, f['title'])
        if f['mimeType'].split('.')[-1] == 'folder':  # folder
            # create folder
            if not os.path.isdir(f_path):
                os.mkdir(f_path)
            # fill contents
            for child_id in self._list_folder(fileid):
                self.download(child_id, directory=f_path)
        else:  # single file
            # check if file exists
            if os.path.isfile(f_path):
                remove = False
                statinfo = os.stat(f_path)
                # filesize different
                if not int(statinfo.st_size) == int(f['fileSize']):
                    remove = True
                # modified since creation
                modified_date_str = f['modifiedDate'].split('.')[0]
                modified_stamp = time.mktime(datetime.datetime.strptime(modified_date_str, '%Y-%m-%dT%H:%M:%S').timetuple())
                if statinfo.st_mtime < modified_stamp:
                    remove = True
                # overwrite toggle
                if overwrite:
                    remove = True
                # remove
                if remove:
                    os.remove(f_path)
                else:
                    return f
            # download
            f.GetContentFile(f_path)
            if verbose:
                print 'file downloaded to {}'.format(f_path)
            # finish
            return f
            
    def list_folder(self, *args, **kwargs):
        return self._list_folder(*args, **kwargs)
            