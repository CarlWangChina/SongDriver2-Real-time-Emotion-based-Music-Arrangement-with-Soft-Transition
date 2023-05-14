// 歌曲改编

import { useEffect, useState } from "react";
import styles from "./index.module.css";
import MIDI from "midi.js";
import { useLocation, useNavigate } from "react-router-dom";
import inputoForModel from "../../assets/inputForModel.json";
import btnTxt from "../../assets/btnTxt.json";

// 这部分的话得结合织体代码进行更改，播放结束后将所有信息上报给后端后，调用navigate("/information")跳转到酬金搜集表格
let timer = null;

window.worker = null;
function midi_contextInit(callback) {
  if (window.worker != null) {
    return;
  }
  //midi环境的回调函数
  MIDI.loadPlugin({
    soundfontUrl: "/soundfont/",
    instruments: ["acoustic_grand_piano"],
    onprogress: function (state, progress) {},
    onsuccess: function () {
      if (window.worker != null) {
        return;
      }
      window.worker = new Worker("chordScriptWorker.js");
      worker.onmessage = function (ev) {
        if (ev.data.m == "onPlayNote") {
          if (ev.data.vel <= 0) {
            MIDI.noteOff(ev.data.channel, ev.data.tone, 0);
          } else {
            MIDI.noteOn(ev.data.channel, ev.data.tone, ev.data.vel, 0);
          }
        } else if (ev.data.m == "onSetIns") {
          MIDI.channels[ev.data.channel].instrument = ev.data.instrumentId;
        } else if (ev.data.m == "ready") {
          callback();
        }
      };
    },
  });
}

//调用此函数选择织体
//例如 midi_useScript("scripts/sleepy.lua");
function midi_useScript(name) {
  worker.postMessage({ m: "useScript", src: name });
}

//这两个控制播放和暂停
function midi_start() {
  worker.postMessage({ m: "start" });
}
function midi_stop() {
  worker.postMessage({ m: "stop" });
}
const gatheringType = {
  INIT: 0, // 配置初始化
  TRUE: 1, // 可以进行上报
  FALSE: 2, // 禁止上报
};

export default () => {
  const [count, setCount] = useState(0);
  const [gathering, setGathering] = useState(gatheringType.INIT); // 是否可以上报

  useEffect(() => {
    //等midi_contextInit回调触发后才允许操作页面
    //不知为什么，调用这个会卡死浏览器
    midi_contextInit(function () {
      setGathering(gatheringType.TRUE);
      console.log("ready");
    });

    console.log("play");
    const model = location.state?.model; // 模型号
    const music = location.state?.music; // 音乐名
    if (!model || !music) return; // TODO，如果没有模型号或音乐名则需要进行异常处理

    // TODO, 在这里发送请求获取音乐数据，目前这里是拿了个mid文件作为示例
    MIDI.Player.loadFile("/test.mid", () => {
      MIDI.Player.addListener((data) => {
        const now = data.now; // where we are now
        const end = data.end; // time when song ends
        console.log(now);
      });
      MIDI.Player.start();
      MIDI.Player.pause();
    });
    return MIDI.Player.stop;
  }, []);

  // 选择情感标签
  function chooseEmotion(e) {
    const emotion = e?.target?.dataset?.emotion;
    const data = inputoForModel[emotion];
    console.log(data);
    if (!data) return; // data是情感标签对应的模型输入
    // TODO,在这里发送网络请求上报并获取接下来的音乐字符串
    worker.postMessage({ m: "toServer", str: "emotion:" + emotion });

    // TODO, 以下操作需要在上报完成的回调中完成
    setCount(8);
    timer = setInterval(() => {
      setCount((prev) => {
        if (prev - 1 === 0) {
          clearInterval(timer);

          setGathering(true); // 开启上报
          MIDI.Player.pause(); // 暂停音乐播放
        }
        return prev - 1;
      });
    }, 1000);
    setGathering(false);
    MIDI.Player.resume(); // 继续播放音乐
  }

  return (
    <div className={styles.adaptation}>
      {gathering === gatheringType.TRUE ? (
        "选择你当前的情感"
      ) : gathering === gatheringType.INIT ? (
        "系统正在初始化中，请稍后"
      ) : (
        <div
          style={{
            fontSize: "xx-large",
            fontWeight: "bolder",
            color: count > 3 ? "black" : "red",
          }}
        >
          {count + " s"}
        </div>
      )}

      <div
        className={styles.myBtns}
        style={{ pointerEvents: gathering ? "all" : "none" }}
        onClick={chooseEmotion}
      >
        {/* 按钮组 */}
        {btnTxt.map((btns, index) => {
          return (
            <div key={index} className={styles.btnBox}>
              {btns.map((item) => {
                return (
                  <div className={styles.btn} data-emotion={item} key={item}>
                    {item}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
};
