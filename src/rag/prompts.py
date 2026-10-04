"""
Prompt 模板模块 — 卫星异常诊断专用

设计原则：
- 系统提示词丰富：给足专业背景、通道知识、异常分类、输出约束
- 输出结果干练：强制结构化、限制条数、禁止废话、一目了然
"""

from typing import Dict, Any, Optional
import yaml
import logging
from pathlib import Path

# 配置日志（由应用入口统一配置 basicConfig）
logger = logging.getLogger(__name__)


class PromptTemplates:
    """Prompt 模板管理器 — 系统提示词丰富，输出强制精简"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.prompt_config = self.config.get("prompt", {})

        # ── 系统提示词：丰富专业背景 + 硬性格式约束 ──
        self.default_system_prompt = """你是OPS-SAT微小卫星遥测异常诊断专家，面向地面测控站运维人员。

## 专业知识
- OPS-SAT：ESA立方星实验平台，搭载磁力计、光电二极管、温度/电流电压传感器
- 磁力计 3 路（各通道与官方名一一对应，**不要自行编号**）：
  CADC0872 = 磁力计0号 = I_B_FB_MM_0（1s 采样，量级 ~1e-5）
  CADC0873 = 磁力计1号 = I_B_FB_MM_1（1s 采样，量级 ~1e-5）
  CADC0874 = 磁力计2号 = I_B_FB_MM_2（1s 采样，量级 ~1e-5）
- 光电二极管 6 路测角（太阳敏感器，5s 采样，值域 [0, π/2] 弧度）：
  CADC0884 = I_PD1_THETA；CADC0886 = I_PD2_THETA；CADC0888 = I_PD3_THETA
  CADC0890 = I_PD4_THETA；CADC0892 = I_PD5_THETA；CADC0894 = I_PD6_THETA
- 异常类型：数值跳变、持续偏离、周期性波动、噪声增大、数据缺失、尖峰脉冲

## 事实边界（不得越界推断）
- **磁力计官方只给编号 I_B_FB_MM_0/1/2，未定义轴向。** 不得推断
  各通道的轴向，也不得在输出里写任何具体轴名。只能照抄上面的「磁力计N号」。
- 同理，光电二极管官方名为 I_PD*_THETA（测角），但**未说明各路对应哪个姿态轴**，
  不得臆断。
- **引用通道名时逐字照抄上面的对应关系**，不要根据通道数字自行推算
  （例如 CADC0874 是 I_B_FB_MM_2，不是 MM_3）。
- 若参考资料未提供某项事实，**直接说「资料未提供」，不要用推测填补**。
  编造一个看似合理的结论比承认不知道危害更大。

## 常见模式
- 磁力计异常+姿态偏差 → 磁力矩器故障/外部磁干扰
- 光电二极管异常+单轴偏离 → 太阳敏感器遮挡/姿态控制问题
- 多通道同时异常 → 电源/总线故障/地磁暴/进出阴影
- 周期性异常 → 轨道周期相关（地影/温度循环）

## 输出硬约束（违反则不合格）
1. 固定7段结构，每段用 **【标题】** 粗体独占一行，格式如下：

**【通道定位】**
（内容）

**【异常类型】**
（内容）

**【可能原因】**
1. 原因一
2. 原因二
3. 原因三

**【影响评估】**
1. 影响一
2. 影响二

**【建议措施】**
1. 建议一
2. 建议二

**【紧急程度】**
低/中/高/紧急（仅一个词）

**【来源】**
1. [文档名 p.页码]

2. 【可能原因】最多3条，降序，每条≤30字
3. 【影响评估】【建议措施】各最多2条，每条≤30字
4. 【紧急程度】仅一个词：低/中/高/紧急
5. 【来源】格式：[文档名 p.页码]，无依据写"知识库未覆盖"
6. 禁止：寒暄、解释性废话、"综上所述"、emoji
7. 中文，工程师语言，一条一行
8. 必须完整输出所有7个section，禁止截断，禁止省略【建议措施】和【来源】"""

        # ── 默认用户模板：结构化，给足上下文 ──
        self.default_user_template = """请对以下遥测异常进行诊断。

【异常信息】
- 通道: {channel_id}
- 异常类型: {anomaly_type}
- 异常描述: {anomaly_description}

【参考知识片段】
{context}

请严格按照系统提示中的结构输出诊断结论。"""

        self.default_citation_format = "[{document_name} p.{page_number}]"

        # 注意：必须用 `or` 而非 dict.get 的默认值参数。
        # dict.get(key, default) 仅在"键不存在"时返回 default；
        # 当键存在但值为空串时（如 system_prompt: ""），get 会返回空串，
        # 导致精心设计的默认提示词被静默吞掉，且下游 `if system_prompt:` 判定为假，
        # 连 system 消息都不会发送给模型。此处统一改为"空值也回落"。
        self.system_prompt = self.prompt_config.get("system_prompt") or self.default_system_prompt
        self.user_template = self.prompt_config.get("user_template") or self.default_user_template
        self.citation_format = self.prompt_config.get("citation_format") or self.default_citation_format

        logger.info("Prompt 模板管理器初始化完成")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "rag_config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"配置加载成功：{config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置未找到：{config_path}，使用默认")
            return {}
        except Exception as e:
            logger.error(f"配置加载失败：{e}")
            return {}

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def get_user_prompt(
        self,
        channel_id: str = "",
        anomaly_type: str = "",
        anomaly_description: str = "",
        context: str = "",
        **kwargs
    ) -> str:
        template_vars = {
            "channel_id": channel_id,
            "anomaly_type": anomaly_type,
            "anomaly_description": anomaly_description,
            "context": context,
            **kwargs
        }
        try:
            return self.user_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"模板变量缺失：{e}，使用简化版")
            return self._get_simplified_prompt(context, **kwargs)

    def _get_simplified_prompt(self, context: str, **kwargs) -> str:
        prompt = "请对以下遥测异常进行诊断。\n\n"
        if kwargs:
            for k, v in kwargs.items():
                if v:
                    prompt += f"- {k}: {v}\n"
        prompt += f"\n参考知识:\n{context}\n\n"
        prompt += "按【通道定位】【异常类型】【可能原因】【影响评估】【建议措施】【紧急程度】【来源】结构输出。"
        return prompt

    def get_rag_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        system = system_prompt or self.system_prompt
        user = (
            f"请对以下遥测异常进行诊断。\n\n"
            f"【异常信息】\n{query}\n\n"
            f"【参考知识片段】\n{context}\n\n"
            "请严格按照系统提示中的7段结构输出诊断结论。"
        )
        return {"system": system, "user": user}

    def get_anomaly_analysis_prompt(
        self,
        channel_id: str,
        anomaly_type: str,
        anomaly_description: str,
        context: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        system = self.system_prompt

        user = f"""请对以下遥测异常进行诊断。

【异常信息】
- 通道: {channel_id}
- 异常类型: {anomaly_type}
- 异常描述: {anomaly_description}"""

        if additional_info:
            user += "\n\n【补充信息】\n"
            for k, v in additional_info.items():
                user += f"- {k}: {v}\n"

        user += f"""

【参考知识片段】
{context}

请严格按照系统提示中的结构输出诊断结论。"""
        return {"system": system, "user": user}

    def get_comparison_prompt(
        self,
        query: str,
        context: str,
        comparison_type: str = "similar"
    ) -> Dict[str, str]:
        system = self.system_prompt

        type_map = {
            "similar": "相似点",
            "different": "差异点",
            "evolution": "发展演变"
        }
        label = type_map.get(comparison_type, "分析")

        user = (
            f"对比分析：{query}\n\n"
            f"参考知识:\n{context}\n\n"
            f"要求：列出{label}，最多3点，每点一行，每点≤30字，禁止废话。"
        )
        return {"system": system, "user": user}

    def format_citation(self, document_name: str, page_number: int) -> str:
        return self.citation_format.format(
            document_name=document_name,
            page_number=page_number
        )

    def update_templates(self, **kwargs):
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]
        if "user_template" in kwargs:
            self.user_template = kwargs["user_template"]
        if "citation_format" in kwargs:
            self.citation_format = kwargs["citation_format"]
        logger.info("模板配置已更新")

    def get_all_templates(self) -> Dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "citation_format": self.citation_format
        }

    def __repr__(self) -> str:
        return f"PromptTemplates(templates={len(self.get_all_templates())})"


# 全局实例（单例模式）
_global_templates = None

def get_prompt_templates(config_path: Optional[str] = None) -> PromptTemplates:
    global _global_templates
    if _global_templates is None:
        _global_templates = PromptTemplates(config_path)
    return _global_templates


def get_system_prompt() -> str:
    return get_prompt_templates().get_system_prompt()


def get_user_prompt(**kwargs) -> str:
    return get_prompt_templates().get_user_prompt(**kwargs)


if __name__ == "__main__":
    print("=== Prompt 模板测试 ===")
    templates = PromptTemplates()
    print(f"模板管理器：{templates}")
    print(f"\n系统提示词长度：{len(templates.get_system_prompt())} 字")
    user_prompt = templates.get_user_prompt(
        channel_id="CADC0874",
        anomaly_type="数值异常",
        anomaly_description="遥测值超出正常范围",
        context="卫星遥测数据..."
    )
    print(f"\n用户提示词：\n{user_prompt[:300]}...")
    print(f"\n来源标注：{templates.format_citation('NASA SOA 2024', 42)}")
    print("\n测试完成！")
