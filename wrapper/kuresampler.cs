using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

/// <summary>
/// 同じフォルダにある (basename)_child.cmd または (basename)_child.bat を呼び出す。
/// </summary>
class Program
{
    static int Main(string[] args)
    {
        try
        {
            // このプログラム自身のパスを取得
            string exePath = Process.GetCurrentProcess().MainModule.FileName;
            string dir = Path.GetDirectoryName(exePath);
            string nameWithoutExt = Path.GetFileNameWithoutExtension(exePath);

            // 子プロセス用のファイルを探す (.cmd → .bat の順)
            string[] extensions = new string[] { ".cmd", ".bat" };
            string childPath = null;

            foreach (string ext in extensions)
            {
                string candidatePath = Path.Combine(dir, nameWithoutExt + "_child" + ext);
                if (File.Exists(candidatePath))
                {
                    childPath = candidatePath;
                    break;
                }
            }

            // 子プロセスが見つからない場合はエラー
            if (childPath == null)
            {
                Console.Error.WriteLine(string.Format("Error: {0}_child.cmd/.bat が見つかりません", nameWithoutExt));
                return 2;
            }

            // 引数をエスケープ（スペースやダブルクォート対応）
            string arguments = string.Join(" ", args.Select(new Func<string, string>(EscapeArg)));
            // Console.WriteLine(arguments); // デバッグ用: 必要なら有効化

            // プロセス情報を準備（パスにスペースが含まれていても良いように cmd.exe を経由せずバッチを直接呼ぶ）
            var startInfo = new ProcessStartInfo();
            startInfo.WorkingDirectory = Environment.CurrentDirectory; // 呼び出し元のCWDを維持
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = false;
            // startInfo.FileName = "cmd.exe"; // cmd.exe 経由で呼び出す場合
            // startInfo.Arguments = string.Format("/c \"{0}\" {1}", childPath, arguments); // cmd.exe 経由で呼び出す場合
            startInfo.FileName = childPath; // 直接呼び出しの場合
            startInfo.Arguments = arguments; // 直接呼び出しの場合

            // 子プロセスを起動して完了を待機
            Process process = Process.Start(startInfo);
            if (process == null)
            {
                Console.Error.WriteLine(string.Format("Error: プロセスの起動に失敗しました ({0})", childPath));
                return 3;
            }

            process.WaitForExit();
            int exitCode = process.ExitCode;
            process.Dispose();
            return exitCode;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(string.Format("Error in {0}: {1}", "Main", ex.Message));
            return 1;
        }
    }

    static string EscapeArg(string arg)
    {
        return "\"" + arg.Replace("\"", "\\\"") + "\"";
    }
}
