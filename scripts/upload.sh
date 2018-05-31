#!/bin/bash
cd "$(dirname "$0")"
cd ..
# test
if [ -z "$PYTHON" ];
then
    export PYTHON=python3
fi
printf "\033[1;35mtesting...\033[0m\n"
if [ -n $TEST ] || $PYTHON setup.py test
then
    printf "\033[1;35mtesting successful\033[0m\n"
else
    printf "\033[1;35mtests failed---aborting\033[0m\n"
    exit 1
fi
# build
printf "\033[1;35mbuilding...\033[0m\n"
$PYTHON setup.py sdist
$PYTHON setup.py bdist_wheel --universal
# upload
printf "\033[1;35muploading...\033[0m\n"
while true; do
        read -p "upload to PyPI? [y/n]: " yn
        case $yn in
            [Nn]*) printf "\033[1;35maborted\033[0m\n" ; break ;;
            [Yy]*) twine upload dist/* ; break ;;
        esac
    done
# cleanup
printf "\033[1;35mcleaning up...\033[0m\n"
echo removing build; rm -r build
echo removing dist; rm -r dist
echo removing eggs; rm -rf ./*.egg*
printf "\033[1;35m...done\033[0m\n"
