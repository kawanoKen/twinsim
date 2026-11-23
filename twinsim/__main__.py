"""コマンド:
  python -m twinsim spaces                         空間の一覧
  python -m twinsim templates                      テンプレートの一覧
  python -m twinsim validate <シナリオ>             シナリオの確かめ（行き先・カメラの名前など）
  python -m twinsim template <名前> [--out 場所]     テンプレートを JSON で書き出す（設定のひな形。既定を足した全部の値は --resolved）
  python -m twinsim run <シナリオ> [--out 場所] [--set key=value ...]   人の動きを計算して書き出す（描画はしない）
                                                   シナリオは JSON か YAML のファイル、テンプレートの名前
  python -m twinsim check <結果の場所>               結果の確かめ（壁の中・人の近さ・歩く速さ・止まる時間）と文献との比べ（validity.*）
  python -m twinsim export-logs <結果の場所>         ゲートの通過記録・EV の運行記録の形で書き出す（実データの形の見本・往復の確かめ）
  python -m twinsim ward-layout [--space hospital] [--out 場所]   病棟のベッド・ナースステーションの置き方と、その間の道のり
  python -m twinsim calibrate <元のシナリオ> --gate-log a.csv --ev-log b.csv [--ev-mode replay]   記録で値を合わせたシナリオを作る
  python -m twinsim render <結果の場所> [--frames 始まり 終わり] [--cameras A,B] [--chunk-s 秒] [--drop-frames] [--labels] [--preroll コマ] [--cull]
                                                Isaac Sim で描画して video/<カメラ>.mp4 を作る（GPU）。--chunk-s で分けて、動いている受け付け全部で並べて描く
  python -m twinsim export-space <名前> --usd URL --region x0 x1 y0 y1 z0 z1 [--res 0.25]   USD から空間の材料を作る（GPU）
  python -m twinsim serve [--port 8770] [--url]       GUI と REST API（127.0.0.1 だけ、トークンつき）
  python -m twinsim show-space <空間> [--out 図.png]   真上の図に歩ける所・目盛り・候補・場所・ゾーンを重ねた図（spaces/<名前>/annotate.png）。
                                                   空間は名前・フォルダ・YAML か JSON のファイル。真上の図のない空間は、歩ける所の格子から平面図を描く
  python -m twinsim bev-detect <結果の場所> [--detector gt_boxes|peoplenet|external] [--gpu 番号] [--group 群] [--name 名前] [--source 場所] [--miss 0.1 ...]
                                                各カメラの人の枠（bev_dets/<名前>/）。gt_boxes は正解の枠から（GPU なし）
  python -m twinsim bev-track <結果の場所> [--tracker a] [--group 群] [--dets 検出] [--preset 名前] [--param key=value ...] [--label 名前]
                                                群ごとの床の上の追跡（bev_tracks/<名前>/<群>/）
  python -m twinsim bev-eval <結果の場所> [--tracker a] [--group 群] [--frames 始まり 終わり] [--vis-min 0.4] [--portal-s 1.0]
                                                床の上の追跡を正解と比べる（bev_eval/）
--set の例: --set duration_s=60 --set seed=3 --set streams.0.rate.per_min=20
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__, api, config
from .scenario import TEMPLATES, deep_merge, load_config, load_scenario, read_config
from .space import list_spaces


def parse_set(items):
    over = {}
    for it in items or []:
        k, v = it.split("=", 1)
        v = yaml.safe_load(v)
        cur = over
        keys = k.split(".")
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = v
    return over


def main(argv=None):
    try:
        return cli(sys.argv[1:] if argv is None else argv)
    except (ValueError, FileNotFoundError) as e:  # 設定の誤りは、スタックトレースなしで知らせる
        print(e, file=sys.stderr)
        return 1


def cli(argv):
    if argv[:1] == ["serve"]:
        from .server import main as serve
        return serve(argv[1:])
    ap = argparse.ArgumentParser(prog="twinsim")
    ap.add_argument("--version", action="version", version=f"twinsim {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spaces")
    sub.add_parser("templates")
    tp = sub.add_parser("template")
    tp.add_argument("name")
    tp.add_argument("--resolved", action="store_true", help="既定の値を足した全部の値を書き出す")
    tp.add_argument("--out", help="書く場所（既定は標準出力）")
    v = sub.add_parser("validate")
    v.add_argument("scenario")
    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--out", default=None)
    r.add_argument("--set", action="append")
    c = sub.add_parser("check")
    c.add_argument("run_dir")
    ge = sub.add_parser("group-eval", help="グループを見つける課題の評価（twinsim/group_eval.py）")
    ge.add_argument("run_dir")
    ge.add_argument("found", help="見つけた結果の JSON（{\"frames\": {コマ: [[人, ...], ...]}}）。perfect で正解そのもの（確かめ）")
    rd = sub.add_parser("render")
    rd.add_argument("run_dir")
    rd.add_argument("--frames", nargs=2, type=int)
    rd.add_argument("--cameras")
    rd.add_argument("--preset", default="RAY_TRACE")
    rd.add_argument("--chunk-s", type=float)
    rd.add_argument("--drop-frames", action="store_true")
    rd.add_argument("--labels", action="store_true", help="描画から人ごとの枠と隠れの割合を出す（labels/<カメラ>.csv）")
    rd.add_argument("--preroll", type=int, default=0, help="区間ごとに前を余分に描いて捨てるコマの数")
    rd.add_argument("--cull", action="store_true", help="カメラごとの区間で、そのカメラの近くの人だけを置く（--chunk-s のとき）")
    ss = sub.add_parser("show-space")
    ss.add_argument("name")
    ss.add_argument("--out", help="図の場所（省くと空間のフォルダの annotate.png）")
    el = sub.add_parser("export-logs")
    el.add_argument("run_dir")
    el.add_argument("--out")
    wl = sub.add_parser("ward-layout")
    wl.add_argument("--space", default="hospital")
    wl.add_argument("--out")
    ca = sub.add_parser("calibrate")
    ca.add_argument("base", help="元のシナリオ（テンプレートの名前かファイル）")
    ca.add_argument("--gate-log")
    ca.add_argument("--ev-log")
    ca.add_argument("--ev-mode", choices=["fit", "replay"], default="fit")
    ca.add_argument("--out", help="書く場所（既定 <TWINSIM_DATA>/calibration/<id>/）")
    ca.add_argument("--id")
    ex = sub.add_parser("export-space")
    ex.add_argument("name")
    ex.add_argument("--usd", required=True)
    ex.add_argument("--region", nargs=6, type=float, required=True)
    ex.add_argument("--res", type=float, default=0.25)
    bd = sub.add_parser("bev-detect")
    bd.add_argument("run_dir")
    bd.add_argument("--detector", default="gt_boxes")
    bd.add_argument("--group", action="append", help="群（何度でも。既定は全部）")
    bd.add_argument("--name", help="bev_dets/ の下の名前（既定は検出器の名前）")
    bd.add_argument("--source", help="external: 同じ形の CSV（<カメラ>.txt）のフォルダ")
    bd.add_argument("--vis-min", type=float, default=0.4)
    bd.add_argument("--jitter-px", type=float, default=0.0)
    bd.add_argument("--miss", type=float, default=0.0)
    bd.add_argument("--fp-per-frame", type=float, default=0.0)
    bd.add_argument("--seed", type=int, default=0)
    bd.add_argument("--gpu", type=int, help="peoplenet: 使う GPU の番号（nvidia-smi で空きを見てから）")
    bd.add_argument("--models", help="peoplenet: モデルの置き場所（既定は TWINSIM_DEEPSTREAM_MODELS）")
    bd.add_argument("--image", help="peoplenet: DeepStream のコンテナ（既定は TWINSIM_DEEPSTREAM_IMAGE）")
    bd.add_argument("--keep-raw", action="store_true", help="peoplenet: DeepStream の KITTI の出力を消さずに残す")
    bt = sub.add_parser("bev-track")
    bt.add_argument("run_dir")
    bt.add_argument("--tracker", default="a")
    bt.add_argument("--group", action="append")
    bt.add_argument("--dets", default="gt_boxes")
    bt.add_argument("--preset", help="twinsim/bev/presets/<名前>.json（A の既定は default）")
    bt.add_argument("--param", action="append", help="設定を上書き（key=value）")
    bt.add_argument("--label", help="bev_tracks/ の下の名前（既定は追跡器の名前）")
    be = sub.add_parser("bev-eval")
    be.add_argument("run_dir")
    be.add_argument("--tracker", default="a", help="bev_tracks/ の下の名前")
    be.add_argument("--group", action="append")
    be.add_argument("--frames", nargs=2, type=int, help="評価する動画のコマ [始まり, 終わり)")
    be.add_argument("--vis-min", type=float, default=0.4, help="群のどのカメラでもこれ未満しか見えない人は判定しない（負の数で使わない）")
    be.add_argument("--portal-s", type=float, default=1.0, help="評価領域の中で現れた・消えた人の前後この秒は判定しない（0 で使わない）")
    be.add_argument("--kind", default="online", choices=["online", "corrected"])
    be.add_argument("--label", help="bev_eval/ の結果の名前（既定は --tracker）")
    a = ap.parse_args(argv)
    if a.cmd == "spaces":
        print("\n".join(list_spaces()))
    elif a.cmd == "templates":
        for p in sorted(TEMPLATES.glob("*.yaml")):
            s = yaml.safe_load(p.read_text(encoding="utf-8"))
            print(f"{p.stem:24s} {s.get('space', ''):10s} {s.get('title', '')}")
    elif a.cmd == "template":
        text = json.dumps(load_config(a.name) if a.resolved else read_config(a.name), ensure_ascii=False, indent=2)
        if a.out:
            Path(a.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    elif a.cmd == "validate":
        spec = load_scenario(a.scenario)
        sp_name = (spec["space"].get("name", "custom") + "（dict）") if isinstance(spec["space"], dict) else spec["space"]
        print("ok:", spec.get("name"), "space", sp_name, "streams", len(spec["streams"]), "cameras", [c["name"] for c in spec["cameras"]])
    elif a.cmd in ("render", "export-space"):
        from . import isaac_bridge as ib
        ib.sync()
        if not ib.ready():
            print("Isaac Sim が動いていない。先に ./isaac_up.sh か tools/render_farm.sh up を実行する（終わったら isim down・render_farm.sh down）")
            return 1
        if a.cmd == "render":
            r = ib.render(a.run_dir, a.frames, a.cameras.split(",") if a.cameras else None, a.preset,
                          chunk_s=a.chunk_s, keep_frames=not a.drop_frames, labels=a.labels, preroll=a.preroll, cull=a.cull)
        else:
            r = ib.export_space(a.name, a.usd, a.region, a.res)
        print(r["log"])
        print(json.dumps({k: v for k, v in r.items() if k != "log"}, ensure_ascii=False))
    elif a.cmd == "show-space":
        from .spaceview import draw
        print(draw(a.name, out=a.out))
    elif a.cmd == "export-logs":
        from .calibrate import export_logs
        print("logs:", export_logs(a.run_dir, a.out))
    elif a.cmd == "ward-layout":
        from .ward import write
        print("ward layout:", write(a.space, a.out))
    elif a.cmd == "calibrate":
        from .calibrate import calibrate
        path, rep = calibrate(a.base, a.gate_log, a.ev_log, a.out, a.ev_mode, cid=a.id)
        for c in rep["changes"]:
            print("合わせた:", json.dumps(c, ensure_ascii=False)[:200])
        for c in rep["kept"]:
            print("元のまま:", json.dumps(c, ensure_ascii=False))
        print("scenario:", path)
    elif a.cmd == "group-eval":
        from . import group_eval
        found = group_eval.perfect(a.run_dir) if a.found == "perfect" else a.found
        print(json.dumps(group_eval.evaluate(a.run_dir, found), ensure_ascii=False, indent=1))
    elif a.cmd == "check":
        from .quality import check
        from .validity import evaluate
        print(json.dumps(check(a.run_dir), ensure_ascii=False, indent=1))
        v = evaluate(a.run_dir)
        for part in ("input_reproduction", "emergent"):
            for k, x in v[part].items():
                items = x.items() if k == "arrivals" else [(k, x)]
                for name, y in items:
                    print(f"{part:18s} {name:28s} {y.get('verdict', '-'):4s} {y.get('note', '')}")
        print("validity:", Path(a.run_dir).expanduser() / "validity.md")
    elif a.cmd == "bev-detect":
        from .bev import detect
        opts = ({"vis_min": a.vis_min, "jitter_px": a.jitter_px, "miss": a.miss, "fp_per_frame": a.fp_per_frame} if a.detector == "gt_boxes" else
                {"gpu": a.gpu, "models": a.models, "image": a.image, "keep_raw": a.keep_raw} if a.detector == "peoplenet" else {})
        out, info = detect(a.run_dir, a.detector, a.group, name=a.name, source=a.source, seed=a.seed, **opts)
        print(json.dumps({k: info[k] for k in ("detector", "cameras", "boxes", "elapsed_s")}, ensure_ascii=False))
        print("out:", out)
    elif a.cmd == "bev-track":
        from .bev import track
        params = {k: yaml.safe_load(v) for k, v in (it.split("=", 1) for it in a.param or [])}
        for g, info in track(a.run_dir, a.tracker, a.group, a.dets, a.preset, params, a.label).items():
            print(json.dumps({"group": g, **{k: info[k] for k in ("tracker", "preset", "points", "ids", "ms_per_frame")}}, ensure_ascii=False))
    elif a.cmd == "bev-eval":
        from .bev import evaluate
        frames = range(*a.frames) if a.frames else None
        res = evaluate(a.run_dir, a.tracker, a.group, vis_min=None if a.vis_min < 0 else a.vis_min, portal_s=a.portal_s, frames=frames,
                       kind=a.kind, label=a.label)
        for g, r in res.items():
            for thr, m in r["metrics"].items():
                print(f"{r['label']:12s} {g:10s} {thr:5s} " + " ".join(f"{k} {m[k]:.3f}" if isinstance(m[k], float) else f"{k} {m[k]}"
                                                                    for k in ("IDF1", "HOTA", "MOTA", "MODA", "IDSW", "Prec", "Rec")))
        print("out:", Path(a.run_dir).expanduser() / "bev_eval")
    elif a.cmd == "run":
        over = parse_set(a.set)
        streams = over.pop("streams", {})
        spec = load_scenario(a.scenario, over)  # people.profile の順の決まりどおりに重ねる
        for idx, sub_over in streams.items():  # streams.0.x のような番号つきの指定は、読んだシナリオに直接当てる
            spec["streams"][int(idx)] = deep_merge(spec["streams"][int(idx)], sub_over)
        out = a.out or str(config.RUNS / (spec.get("name") or Path(a.scenario).stem))
        summary = api.run(spec, out)
        print(json.dumps({k: summary[k] for k in ("people", "groups", "max_present", "elevator", "gate_passes", "timing_s")},
                         ensure_ascii=False, default=str))
        print("out:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
