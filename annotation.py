#!/usr/bin/python3
import os
import argparse
import copy
from PIL import Image
from jinja2 import Template
import xmltodict




class Main:

        def __init__(self, args):
                self._args = args
                self._ANNOTATION = {
                        "folder":"TBD",
                        "filename":"TBD",
                        "path":"TBD",
                        "size":{
                                "width":-1,
                                "height":-1,
                                "depth":3
                        },
                        "object":{
                                "name":"TBD",
                                "pose":"Unspecified",
                                "truncated":0,
                                "difficult":0,
                                "bndbox":{
                                        "xmin":-123,
                                        "ymin":-123,
                                        "xmax":-123,
                                        "ymax":-123
                                }
                        }
                }




        def start(self):
                template_name =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'annotation.xml.j2')
                with open(template_name) as f:
                        TEMPLATE = Template(f.read())

                files = Main.getFiles(self._args.images)
                for f in files:
                        annotation = copy.deepcopy(self._ANNOTATION)
                        annotation['folder'] = os.path.dirname(f)
                        annotation['filename'] = os.path.basename(f)
                        annotation['path'] = f
                        annotation['object']['name'] = os.path.basename(os.path.dirname(f))
                        w, h = Main.getImageResolution(f)
                        annotation['size']['width'] = w
                        annotation['size']['height'] = h
                        dst = os.path.join(annotation['folder'], annotation['filename']) + '.xml'

                        # Preserving bounding box if xml already exists
                        if os.path.exists(dst):
                                with open(dst) as d:
                                        doc = xmltodict.parse(d.read())
                                xmin = doc['annotation']['object']['bndbox']['xmin']
                                ymin = doc['annotation']['object']['bndbox']['ymin']
                                xmax = doc['annotation']['object']['bndbox']['xmax']
                                ymax = doc['annotation']['object']['bndbox']['ymax']
                                annotation['object']['bndbox']['xmin'] = xmin
                                annotation['object']['bndbox']['ymin'] = ymin
                                annotation['object']['bndbox']['xmax'] = xmax
                                annotation['object']['bndbox']['ymax'] = ymax

                        tr = TEMPLATE.render(annotation=annotation)

                        with open(dst, "w") as d:
                                d.write(tr)

                        if self._args.verbosity > 0:
                                print(f'processing "{f}", writing to "{dst}"')




        @staticmethod
        def getImageResolution(filename):
                with Image.open(filename) as img:
                        width, height = img.size
                return width, height




        @staticmethod
        def getFiles(source):
                res = []
                for root, files in os.walk(source):
                        for file in files:
                                if file.endswith('.jpg'):
                                        f = os.path.join(root, file)
                                        res = res + [f]
                return res




if __name__ == "__main__":
        parser = argparse.ArgumentParser(prog='"annotation.py',
                                         description='Create annotatin XML files for jpg images')
        parser.add_argument('images', help='directory with jpg images')
        parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1], default=0, help="set output verbosity level")
        args = parser.parse_args()

        main = Main(args)
        main.start()
