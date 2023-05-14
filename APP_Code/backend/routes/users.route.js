var express = require("express");
var router = express.Router();

// 存储用户数据
const userData = {};

/* GET users listing. */
router.get("/", function (req, res, next) {
  if (!req.session) {
    return;
  }
  console.log(JSON.parse(req.query.data));
  res.send("respond with a resource");
});

module.exports = router;
