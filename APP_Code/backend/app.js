var createError = require("http-errors");
var express = require("express");
var path = require("path");
// var cookieParser = require("cookie-parser");
var logger = require("morgan");
var session = require("express-session");
var crypto = require("crypto");

/*************** 路由导入 **************/
var usersRouter = require("./routes/users.route");
var musicRouter = require("./routes/music.route");

/*************** 路由导入 **************/

var app = express();

/***************中间件注册 *************/
// view engine setup
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "ejs");

app.use(logger("dev"));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
// app.use(cookieParser());
app.use(express.static(path.join(__dirname, "public")));

// 注册session，用于每8s一次的数据上报
app.use(
  session({
    secret: "paper",
    resave: false,
    saveUninitialized: true,
    cookie: {
      secure: false,
      maxAge: 10 * 60 * 1000,
    },
    rolling: true,
  })
);

// 创建一个uuid
app.use("/", (req, res, next) => {
  if (!req.session.uuid) {
    req.session.uuid = crypto.randomUUID();
  }
  next();
});

/***************中间件注册 *************/

/*****************注册路由 **************/
app.use("/music", musicRouter);
app.use("/users", usersRouter);

/*****************注册路由 **************/

/************* 异常捕获与处理处理 ***********/
app.use(function (req, res, next) {
  next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get("env") === "development" ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render("error");
});
/************* 异常捕获与处理处理 ***********/

module.exports = app;
