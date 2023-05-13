import { Button } from "antd";
import { useNavigate } from "react-router-dom";

const model = {
  recommend: 0,
  model1: 1,
  model2: 2,
  model3: 3,
  model4: 4,
  model5: 5,
  model6: 6,
  model7: 7,
};

export default () => {
  const navigate = useNavigate();
  return (
    <div
      style={{
        height: "100%",
        padding: "10%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-around",
        alignItems: "center",
      }}
      onClick={(e) => {
        // 事件委托，如果没有model属性则无效触发
        const myModel = e.target?.parentNode?.dataset?.model;
        if (!myModel) {
        } else {
          // 推荐治疗界面
          if (myModel == model["recommend"]) navigate("/recommendation");
          // 改编式治疗界面
          else {
            navigate("/home", {
              state: {
                model: myModel,
              },
            });
          }
        }
      }}
    >
      <Button size="large" type="primary" data-model={model["recommend"]}>
        推荐1
      </Button>

      {Object.keys(model).map((item, index) => {
        if (index === 0) return;
        return (
          <Button
            key={item}
            size="large"
            type="primary"
            data-model={model["model" + index]}
          >
            改编{index}
          </Button>
        );
      })}
    </div>
  );
};
