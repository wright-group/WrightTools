"""
Interact with google drive using the pydrive package.
"""

# Darien Morrow - darienmorrow@gmail.com - dmorrow3@wisc.edu
# Blaise Thompson - blaise@untzag.com


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import datetime


# --- define --------------------------------------------------------------------------------------


directory = os.path.dirname(os.path.abspath(__file__))


# --- ensure google drive creds folder populated --------------------------------------------------


creds_dir = os.path.join(directory, 'temp', 'google drive')
if not os.path.isdir(creds_dir):
    os.mkdir(creds_dir)

mycreds_path = os.path.join(creds_dir, 'mycreds.txt')
if not os.path.isfile(mycreds_path):
    open(mycreds_path, 'a').close()


# --- helper methods ------------------------------------------------------------------------------


def id_to_url(driveid):
    return 'https://drive.google.com/open?id=' + driveid


# --- drive class ---------------------------------------------------------------------------------


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

    def _list_folder(self, *args, **kwargs):
        """
        Legacy. Please use self.list_folder instead!
        - Blaise 2016.05.14
        """
        return self.list_folder(*args, **kwargs)

    def _upload_file(self, filepath, parentid, overwrite=False,
                     delete_local=False, verbose=True):
        self._authenticate()
        title = filepath.split(os.path.sep)[-1]
        # check if remote file already exists
        q = {'q': "'{}' in parents and trashed=false".format(parentid)}
        fs = self.api.ListFile(q).GetList()
        f = None
        for fi in fs:
            # dont want to look at folders
            if 'folder' in fi['mimeType']:
                continue
            if fi['title'] == title:
                print(title, 'found in upload file')
                f = fi
        if f is not None:
            remove = False
            statinfo = os.stat(filepath)
            # filesize different
            if not int(statinfo.st_size) == int(f['fileSize']):
                remove = True
            # modified since creation
            remote_stamp = f['modifiedDate'].split('.')[0]  # UTC
            remote_stamp = time.mktime(datetime.datetime.strptime(
                remote_stamp, '%Y-%m-%dT%H:%M:%S').timetuple())
            local_stamp = os.path.getmtime(filepath)  # local
            local_stamp += time.timezone  # UTC
            if local_stamp > remote_stamp:
                remove = True
            # overwrite toggle
            if overwrite:
                remove = True
            # remove
            if remove:
                f.Trash()
                f = None
        # upload
        if f is None:
            f = self.api.CreateFile({'title': title,
                                     'parents': [{"id": parentid}]})
            f.SetContentFile(filepath)
            f.Upload()
            f.content.close()
            if verbose:
                print('file uploaded from {}'.format(filepath))
        # delete local
        if delete_local:
            os.remove(filepath)
        # finish
        return f['id']

    def create_folder(self, name, parentid):
        """ Create a new folder in Google Drive.

        Attributes
        ----------
        name : string or list of string
            Name of new folder to be created or list of new folders and
            subfolders.
        parentID : string
            Google Drive ID of folder that is to be the parent of new folder.

        Returns
        -------
        string
            The unique Google Drive ID of the bottom-most newly created folder.
        """
        import time
        t = time.time()
        self._authenticate()
        print(time.time() - t, "Authenticate")
        t = time.time()
        # clean inputs
        if isinstance(name, str):
            name = [name]
        # create
        parent = parentid
        for n in name:
            # check if folder with that name already exists
            q = {
                'q': "'{}' in parents and trashed=false and mimeType contains \'folder\'".format(parent)}
            fs = self.api.ListFile(q).GetList()
            found = False
            for f in fs:
                if f['title'] == n:
                    found = True
                    parent = f['id']
                    continue
            if found:
                continue
            # if no folder was found, create one
            f = self.api.CreateFile({'title': n,
                                     "parents": [{"id": parent}],
                                     "mimeType": "application/vnd.google-apps.folder"})
            f.Upload()
            parent = f['id']
            print(time.time() - t, "created", n)
            t = time.time()
        return parent

    def download(self, fileid, directory='cwd', overwrite=False, verbose=True):
        """ Recursively download from Google Drive into a local directory.

        By default, will not re-download if file passes following checks:

        1. same size as remote file

        2. local file last modified after remote file

        Parameters
        ----------
        fileid : str
            Google drive id for file or folder.
        directory : str (optional)
            Local directory to save content into. By default saves to cwd.
        overwrite : bool (optional)
            Toggle forcing file overwrites. Default is False.
        verbose : bool (optional)s
            Toggle talkback. Default is True.

        Returns
        -------
        pydrive.files.GoogleDriveFile
        """
        self._authenticate()
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
                remote_stamp = f['modifiedDate'].split('.')[0]  # UTC
                remote_stamp = time.mktime(datetime.datetime.strptime(
                    remote_stamp, '%Y-%m-%dT%H:%M:%S').timetuple())
                local_stamp = os.path.getmtime(f_path)  # local
                local_stamp += time.timezone  # UTC
                if local_stamp < remote_stamp:
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
                print('file downloaded to {}'.format(f_path))
            # finish
            return f

    def list_folder(self, folderid):
        # adapted from https://github.com/googledrive/PyDrive/issues/37
        # folder_id: GoogleDriveFile['id']
        self._authenticate()
        q = {'q': "'{}' in parents and trashed=false".format(folderid)}
        raw_sub_contents = self.api.ListFile(q).GetList()
        return [i['id'] for i in raw_sub_contents]

    def upload(self, path, parentid, overwrite=False, delete_local=False,
               verbose=True):
        """ Upload local file(s) to Google Drive.

        Parameters
        ----------
        path : str
            Path to local file or folder.
        parentid : str
            Google Drive ID of remote folder.
        overwrite : bool (optional)
            Toggle forcing overwrite of remote files. Default is False.
        delete_local : bool (optional)
            Toggle deleting local files and folders once uploaded. Default is
            False.
        verbose : bool (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        driveid : str
            Google Drive ID of folder or file uploaded
        """
        self._authenticate()
        if os.path.isfile(path):
            return self._upload_file(path, parentid, overwrite=overwrite,
                                     delete_local=delete_local,
                                     verbose=verbose)
        elif os.path.isdir(path):
            top_path_length = len(path.split(os.path.sep))
            for tup in os.walk(path, topdown=False):
                self._authenticate()
                folder_path, _, file_names = tup
                print(folder_path)
                # create folder on google drive
                name = folder_path.split(os.path.sep)[top_path_length - 1:]
                folderid = self.create_folder(name, parentid)
                # upload files
                for file_name in file_names:
                    p = os.path.join(folder_path, file_name)
                    self._upload_file(p, folderid, overwrite=overwrite,
                                      delete_local=delete_local,
                                      verbose=verbose)
                # remove folder
                if delete_local:
                    os.rmdir(folder_path)
            # finish
            return folderid
        else:
            raise Exception('path {0} not valid in Drive.upload'.format(path))
