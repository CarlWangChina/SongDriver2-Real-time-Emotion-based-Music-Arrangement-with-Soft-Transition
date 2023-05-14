const fs = require("fs");
const path = require("path");

const musicList = fs
  .readFileSync(path.join(__dirname, "./musicList.csv"))
  .toString()
  .split("\n")
  .map((item) => {
    const t = item.split(",");
    t[t.length - 1] = parseFloat(t[t.length - 1]);
    t[t.length - 2] = parseFloat(t[t.length - 2]);
    return t;
  });

const musicType = [
  "surprise",
  "excited",
  "happy",
  "contented",
  "relaxed",
  "calm",
  "sad",
  "bored",
  "sleepy",
  "tense",
  "angry",
  "worry",
  "elated",
  "pleased",
  "serene",
  "gloomy",
];

const musicRange = {
  happy: {
    x: [0.2, 0.35],
    y: [0.65, 0.8],
  },
  excited: {
    x: [0.45, 0.6],
    y: [0.4, 0.55],
  },
  surprise: {
    x: [0.65, 0.8],
    y: [0.1, 0.25],
  },
  tense: {
    x: [0.65, 0.8],
    y: [-0.25, -0.1],
  },
  angry: {
    x: [0.45, 0.6],
    y: [-0.6, -0.55],
  },
  worry: {
    x: [0.2, 0.35],
    y: [-0.85, -0.7],
  },
  sleepy: {
    x: [-0.85, -0.7],
    y: [-0.25, -0.1],
  },
  bored: {
    x: [-0.65, -0.5],
    y: [-0.55, -0.4],
  },
  sad: {
    x: [-0.35, -0.2],
    y: [-0.85, -0.7],
  },
  contented: {
    x: [-0.35, -0.2],
    y: [0.65, 0.8],
  },
  relaxed: {
    x: [-0.6, -0.45],
    y: [0.4, 0.55],
  },
  calm: {
    x: [-0.85, -0.7],
    y: [0.1, 0.25],
  },
  elated: {
    x: [0.2, 0.4],
    y: [0.25, 0.45],
  },
  pleased: {
    x: [0.05, 0.25],
    y: [0.05, 0.25],
  },
  serene: {
    x: [-0.15, 0.05],
    y: [-0.15, 0.05],
  },
  gloomy: {
    x: [-0.3, -0.1],
    y: [-0.35, -0.15],
  },
};

const music = {};

musicType.forEach((item) => {
  music[item] = musicList
    .filter((music) => {
      const x = music[1];
      const y = music[2];
      return (
        musicRange[item].x[0] <= x &&
        musicRange[item].x[1] >= x &&
        musicRange[item].y[0] <= y &&
        musicRange[item].y[1] >= y
      );
    })
    .map((music) => {
      return music[0];
    });
});
// 根据情感获取音乐
function getMusicByEmotion(emotion) {
  return music[emotion][parseInt((music[emotion].length - 1) * Math.random())];
}

module.exports = {
  getMusicByEmotion,
  musicType,
};
