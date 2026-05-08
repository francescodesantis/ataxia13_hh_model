check C++ standard library (in sysroot, on Mac is something like MacOSX.sdk):

echo "CXXFLAGS=$CXXFLAGS"
echo "SDKROOT=$SDKROOT"
echo "CXXFLAGS=$CXXFLAGS"

and then:

unset CXXFLAGS
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk
export CONDA_BUILD_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk
