####################################################################################
# This script is to define a class FileManager which is responsible for handling
# file operations including adding a new folder, add a new file, change a file,
# so that we can store the output conveniently.
#
# @author Wang Shuai
# @date 2018.03.01
####################################################################################
import os, shutil
import errno

class FileManager(object):

    def __init__(self, fileroot = '/home'):
        self.fileroot = fileroot



    def is_valid_request(self, path):
        '''check whether the given path(file or dir) is within the specified root path'''
        directory = os.path.abspath(self.fileroot)
        file_path = os.path.abspath(path)

        #return true, if the common prefix of both is equal to directory
        #e.g. /a/b/c/d.rst and directory is /a/b, the common prefix is /a/b
        if os.path.commonprefix([file_path, directory]) == directory:
            return True
        else:
            print "not a valid request: not under the root path"
            return False



    def add_dir(self, dirname):
        ''' create a directory under root dir if it not exist'''

        if not self.is_valid_request(dirname):
            print "error: checking"
            return

        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    print "Error: create a new dir " + dirname
                    raise
        else:
            print "dir already exists: " + dirname

    def add_file(self, path):
        '''create a file under root dir if it not exist'''
        #print os.path.isfile(path)
        if not self.is_valid_request(path):
            print "error: checking"
            return

        # create the dir if not exist
        #print os.path.dirname(path)
        self.add_dir(os.path.dirname(path));
        # create the file if not exist

        if not os.path.exists(path):
            try:
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                file_handle = os.open(path, flags)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    print "Error: something unexpected went rong when creating a new file " + path
                    raise
            else:
                with os.fdopen(file_handle, 'w') as file_obj:
                    # using os.fdopen to convert the handle to an object that acts like a
                    # regular Python file object, and the 'with' context manager means
                    # the file will be automatically closed when we are done with it
                    file_obj.write("")
        else:
            print "file already exists"

    def delete_file(self, path):
        '''delete a file if it exists'''
        if not self.is_valid_request(path):
            print "error: checking"
            return

        if os.path.exists(path):
            os.remove(path)
        else:
            print "the file does not exist"

    def delete_content_of_dir(self, dirname):
        '''delete all files under a specific directory'''
        if not self.is_valid_request(dirname) or not os.path.isdir(dirname):
            print "error: checking"
            return

        for the_file in os.listdir(dirname):
            file_path = os.path.join(dirname, the_file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    print "other cases for deleting contents under a dir"
            except Exception as exc:
                print(exc)



if __name__ == "__main__":
    m = FileManager("/home/wshuai/work")
    m.add_file("/home/wshuai/work/test_dir/test.txt")
    m.delete_content_of_dir("/home/wshuai/work")








