module.exports = {
  HOST: "localhost", // 填写mysql的主机名，一般默认值即可
  USER: "root", // 填写mysql的用户名，一般默认值即可
  PASSWORD: "123456", // 填自己的数据库密码
  DB: "newTest", // 填自己的数据库名
  dialect: "mysql", // 这个不动，因为sequelize支持很多种数据库实现，此处选用mysql
};
