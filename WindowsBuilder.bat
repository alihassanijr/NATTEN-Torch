@ECHO off    
if /I %1 == default goto :default
if /I %1 == install goto :install
if /I %1 == clean goto :clean

goto :eof ::can be ommited to run the `default` function similarly to makefiles

:default
echo "Please specify a target."
goto :eof

:install
echo Building NATTEN from source
cmd /V /C "set NATTEN_N_WORKERS=8 && NATTEN_VERBOSE=1 && NATTEN_CUDA_ARCH=8.6 && python setup.py install"
goto :eof

:clean
echo Cleaning up
echo "Removing %CD%\build"
del %CD%\build
goto :eof
