# 解像度別パフォーマンス・ベンチマーク計画書

## 1. プログラムの目的
解像度の違いが「人物検知 + 特徴抽出（ReID）」の処理時間に与える影響を、1004枚×10解像度の計10,040枚の画像を用いて精密に計測し、FPS（1秒あたりの処理枚数）および平均レイテンシを可視化する。

## 2. 実装ファイル
`benchmark_resolution_speed.py` を新規作成します。

## 3. 詳細なアルゴリズム構成

### A. 前準備（Initialization）
*   **Pipelineの初期化**: `person_pipeline.py` の `PersonPipeline` クラスを再利用し、モデル（YOLO + ReID）を1度だけロードします。
*   **計測ディレクトリの走査**: `datasets/resized_samples` 内のフォルダ名を数値としてパースし、昇順（16, 32, ...）にソートして処理対象とします。

### B. 計測プロトコル（Benchmarking Loop）
各解像度において以下の手順を実行します。
1.  **Warm-up**: 最初の5枚を計測対象外として処理させます。これはGPUの初期化やキャッシュの影響を除去するためです。
2.  **Start Timer**: `time.perf_counter()` を呼び出し、ループを開始します。
3.  **Inference Loop**: 1004枚の画像を1枚ずつ読み込み、パイプラインを実行します。
    *   *画像読み込み時間も含めることで、実運用に近いスループットを計測します。*
4.  **End Timer**: 全枚数終了後の時刻を記録します。
5.  **Statistics**: 以下の統計値を算出します。
    *   Total Time (sec)
    *   Average Latency (ms/image)
    *   FPS (frames/sec)
    *   Standard Deviation (ms) : 処理時間のばらつき

### C. 出力成果物（Outputs）
*   **ディレクトリ**: `runs/detect/person_output/benchmark_speed/`
*   **CSVレポート**: `benchmark_results.csv`
    *   カラム: `resolution`, `avg_ms`, `fps`, `std_ms`, `min_ms`, `max_ms`
*   **可視化グラフ**: `speed_analysis.png`
    *   **グラフ1**: 解像度 vs FPS
    *   **グラフ2**: 解像度 vs 平均処理時間（ms）

## 4. 依存ライブラリ
*   `pandas`: データ集計用
*   `matplotlib`: グラフ生成用
*   `tqdm`: プログレスバー表示用
*   `opencv-python`, `torch`, `time`, `statistics`

## 5. 考慮事項
*   **メモリエラーの防止**: 8192x8192などの超高解像度画像はVRAMを大量に消費するため、エラーが発生した場合はスキップし、その旨を記録するように実装します。
*   **計測の純粋性**: 計測中はログ出力（print等）を最小限に抑え、I/O待ち以外のオーバーヘッドを排除します。
