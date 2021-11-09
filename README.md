# 環境
## 仮想環境の作成
python -m venv .venv
## 仮想環境へ切り替え

### コマンド プロンプトでの切り替え
*(作業ディレクトリ)*

.venv\Scripts\activate.bat

### PowerShellでの切り替え
*PowerShellでスクリプトの実行を許可(C直下)*

C:\> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force


*(作業ディレクトリ)*

.venv\Scripts\activate.ps1

## pipでインストールしたもの
opencv-python         4.5.4.58

opencv-contrib-python 4.5.4.58

mediapipe             0.8.9

scikit-learn          1.0.1
