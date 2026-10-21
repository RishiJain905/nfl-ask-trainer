@echo off
rem Build upstream llama.cpp (cloned at tag b11779 into F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream) for the RX 7800 XT with the same settings as the fork's build-rdna3-gfx1101.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\AMD\ROCm\7.1\bin;C:\Users\ytrjx\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1\"
cd /d F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream
cmake -S . -B build-rdna3-gfx1101 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_HIP=ON -DGPU_TARGETS=gfx1101 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_CCACHE=OFF || exit /b 1
cmake --build build-rdna3-gfx1101 --config Release -j 8 || exit /b 2
echo BUILD_OK
