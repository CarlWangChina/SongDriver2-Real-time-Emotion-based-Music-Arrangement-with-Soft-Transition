var express = require("express");
const { getMusicByEmotion, musicType } = require("../utils/recommendMusic");
var router = express.Router();
var fs = require("fs");
const path = require("path");

// TODO，接收用户情感并返回一首歌(推荐式)
router.get("/recommend", (req, res, next) => {
  console.log(req.session.uuid);
  const emotion = req.query.emotion;
  console.log(emotion);
  console.log(musicType.indexOf(emotion));
  if (!emotion || musicType.indexOf(emotion) === -1) {
    res.sendStatus(402);
    return;
  }
  const musicName = getMusicByEmotion(emotion);

  fs.readFile(
    path.join(__dirname, "../public/musicTxt", musicName),
    (err, data) => {
      if (err) {
        console.log(err);
        return;
      }
      // 直接返回文件文本内容
      res.json({
        data: data.toString(),
      });
    }
  );
});
// TODO，接收用户情感，音乐id，进行歌曲改编后返回

module.exports = router;
