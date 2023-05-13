import { Button, Card, Form, Input } from "antd";
const onFinish = (values) => {
  // 在这里上报数据
  console.log("Success:", values);
};
const onFinishFailed = (errorInfo) => {
  console.log("Failed:", errorInfo);
};

const App = () => (
  <Card
    title="酬金信息搜集"
    bordered={false}
    style={{ width: 500, margin: "10% auto" }}
  >
    <Form
      name="basic"
      labelCol={{
        span: 5,
      }}
      wrapperCol={{
        span: 16,
      }}
      style={{
        maxWidth: 600,
      }}
      onFinish={onFinish}
      onFinishFailed={onFinishFailed}
      autoComplete="off"
    >
      <Form.Item
        label="姓名"
        name="name"
        rules={[
          {
            required: true,
            message: "请输入你的姓名!",
          },
        ]}
      >
        <Input />
      </Form.Item>

      <Form.Item
        label="支付宝账号"
        name="account"
        rules={[
          {
            required: true,
            message: "请输入你的支付宝账号，用于后续奖励发放",
          },
        ]}
      >
        <Input />
      </Form.Item>

      <Form.Item
        label="性别"
        name="sex"
        rules={[
          {
            required: true,
            message: "请输入你的性别",
          },
        ]}
      >
        <Input />
      </Form.Item>

      <Form.Item
        label="年龄"
        name="age"
        rules={[
          {
            required: true,
            message: "请输入你的支付宝账号，用于后续奖励发放",
          },
        ]}
      >
        <Input />
      </Form.Item>

      <Form.Item
        wrapperCol={{
          offset: 8,
          span: 16,
        }}
      >
        <Button type="primary" htmlType="submit">
          提交
        </Button>
      </Form.Item>
    </Form>
  </Card>
);
export default App;
