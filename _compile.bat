@rem Compile kuresampler.cs to kuresampler.exe
set selfdir=%~dp0
set sourcefile=kuresampler.cs
set outfile=..\kuresampler.exe
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /nologo /out:"%selfdir%%outfile%" /target:exe /platform:anycpu /optimize+ /warn:4 /utf8output "%selfdir%%sourcefile%"
