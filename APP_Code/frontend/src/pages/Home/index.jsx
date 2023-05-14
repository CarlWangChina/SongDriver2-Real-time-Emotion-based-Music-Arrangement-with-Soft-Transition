// 改编式治疗音乐选择部分

import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { useLocation, useNavigate } from "react-router-dom";
import "./index.css";
import { Input } from "antd";

// 注意，音乐列表一定不能重名
let musicList = [
  "DESPECHA RMX",
  "Face 2 Face",
  "LA FINE",
  "absence",
  "Checkers",
  "The Mud",
  "Nasty",
  "LET GO",
  "Angel",
  "High",
  "Nobody Like You",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "11",
  "12",
  "13",
  "14",
];
let chart = null;
let model = null;
export default () => {
  const location = useLocation(); // 用于获取是第几个模型
  const chartDom = useRef();
  const navigate = useNavigate();
  const [searchRes, setSearchRes] = useState([]);

  // 获取模型
  useEffect(() => {
    model = location?.state?.model;
    console.log(model);
  }, [location]);

  // echarts初始化
  useEffect(() => {
    const chart = initChart(chartDom.current);
    return () => {
      // myChart.dispose() 销毁实例。实例销毁后无法再被使用
      chart.dispose();
    };
  }, []);

  // 初始化图表
  function initChart(canvas) {
    chart = echarts.init(canvas);

    let data = musicList.map((item) => {
      return {
        label: item,
        amount: item.length,
      };
    });

    initBubbleChart(data, ["label", "amount"]);
    return chart;
  }

  // 初始化气泡图
  function initBubbleChart(data = [], format = []) {
    let [maxValue, temp] = [0, []];
    data.forEach((item) => {
      temp.push(item[format[1]]);
    });
    maxValue = Math.max.apply(null, temp);

    // 气泡颜色数组
    let color = [
      "#FFB600",
      "#886CFF",
      "#0084FF",
      "#4CB690",
      "#58B458",
      "#6C6C6C",
      "#F56161",
      "#FC754C",
      "#5F5EEC",
    ];
    // 气泡颜色备份
    let bakeColor = [...color];
    // 气泡数据
    let bubbleData = [];
    // 气泡基础大小
    let basicSize = 70;
    // 节点之间的斥力因子,值越大,气泡间距越大
    let repulsion = 380;
    // 根据气泡数量配置基础大小和斥力因子（以实际情况进行适当调整，使气泡合理分布）
    if (data.length >= 5 && data.length < 10) {
      basicSize = 50;
      repulsion = 230;
    }
    if (data.length >= 10 && data.length < 20) {
      basicSize = 40;
      repulsion = 150;
    } else if (data.length >= 20) {
      basicSize = 30;
      repulsion = 75;
    }

    // 填充气泡数据数组bubbleData
    for (let item of data) {
      // 确保气泡数据条数少于或等于气泡颜色数组大小时，气泡颜色不重复
      if (!bakeColor.length) bakeColor = [...color];
      let colorSet = new Set(bakeColor);
      let curIndex = Math.round(Math.random() * (colorSet.size - 1));
      let curColor = bakeColor[curIndex];
      colorSet.delete(curColor);
      bakeColor = [...colorSet];
      // 气泡大小设置
      let size = (item[format[1]] * basicSize * 3) / maxValue;
      if (size < basicSize) size = basicSize;

      bubbleData.push({
        name: item[format[0]],
        value: item[format[1]],
        symbolSize: size,
        draggable: true,
        itemStyle: {
          normal: { color: curColor },
        },
        label: {
          width: size,
        },
      });
    }

    let bubbleChart = chart;
    // quinticlnOut
    let bubbleOptions = {
      backgroundColor: "#fff",
      animationEasing: "bounceOut",
      animationEasingUpdate: "bounceIn",
      series: [
        {
          type: "graph",
          layout: "force",
          force: {
            repulsion: repulsion,
            edgeLength: 10,
            // initLayout: 'circular',
            friction: 0.4,
          },
          // 是否开启鼠标缩放和平移漫游
          roam: false,
          label: {
            // normal: { show: true },
            show: true,
            color: "#000",
            width: 100,
            overflow: "truncate",
          },
          data: bubbleData,
        },
      ],
    };
    bubbleChart.clear();
    bubbleChart.setOption(bubbleOptions);
    bubbleChart.on("click", (e) => {
      console.log(e.data.value);
      navigate("/adaptation", {
        state: {
          id: e.data.value,
        },
      });
    });
  }

  // 重新获取歌曲
  function resetAll() {
    // TODO，歌曲列表，后续改为从后台获取歌曲列表或者直接将歌曲列表存储到本地然后随机出？这个看你的实现方式
    let musicList = [
      "DESPECHA RMX",
      "Face 2 Face",
      "LA FINE",
      "absence",
      "Checkers",
      "The Mud",
      "Nasty",
      "LET GO",
      "Angel",
      "High",
      "Nobody Like You",
    ];

    // 格式化数据
    let data = musicList.map((item) => {
      return {
        label: item,
        amount: item.length,
      };
    });
    initBubbleChart(data, ["label", "amount"]);
  }

  // 音乐搜索
  function searchMusic(e) {
    const input = e.target.value.toLowerCase();
    if (input.length === 0) {
      setSearchRes([]);
      return;
    }
    const res = musicList.filter((it) => {
      const item = it.toLowerCase();
      let p = 0,
        q = 0;
      for (q = 0; q < input.length; q++) {
        while (p < item.length && item[p++] !== input[q]);
        if (p === item.length) return false;
      }
      return true;
    });
    console.log(res);
    setSearchRes(res);
  }

  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-start",
        padding: "10px",
      }}
    >
      {/* 搜索栏 */}
      <div className="searchBox">
        <Input.Search className="myInput" onInput={searchMusic} />
        {/* 下面是音乐搜索结果框 */}
        <div
          className="searchContent"
          style={{ display: searchRes.length === 0 ? "none" : "block" }}
          onClick={(e) => {
            // 事件委托
            const music = e?.target?.dataset?.music;
            if (!music) return;
            navigate("/adaptation", {
              state: {
                model,
                music,
              },
            });
          }}
        >
          {searchRes.map((item) => {
            return (
              <div key={item} className="myContent" data-music={item}>
                {item}
              </div>
            );
          })}
        </div>
      </div>
      <div
        style={{
          flex: "1",
          width: "100%",
          height: "0",
          display: "flex",
        }}
      >
        <div
          style={{
            height: "100%",
            width: "50%",
          }}
        >
          <h4
            style={{
              width: "100%",
              height: "10%",
              padding: "10px",
            }}
          >
            音乐列表
          </h4>
          <div
            style={{
              height: "85%",
              width: "100%",
              overflow: "auto",
            }}
            onClick={(e) => {
              // 事件委托
              const music = e?.target?.dataset?.music;
              if (!music) return;
              navigate("/adaptation", {
                state: {
                  model,
                  music,
                },
              });
            }}
          >
            {musicList.map((item) => (
              <div key={item} className="listItem" data-music={item}>
                {item}
              </div>
            ))}
          </div>
        </div>

        {/* echarts图表 */}
        <div style={{ height: "100%", width: "50%" }} ref={chartDom}></div>
        <div className="resetBox">
          <div id="reset" className="reset" onClick={resetAll}></div>
          换一批
        </div>
      </div>
    </div>
  );
};
