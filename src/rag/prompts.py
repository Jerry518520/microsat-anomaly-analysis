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

logging.basicConfig(level=logging.INFO)
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
- 磁力计（CADC0872-0874，X/Y/Z）：反映姿态与外部磁场
- 光电二极管（CADC0884-0894，1-6号）：太阳敏感器，用于姿态确定
- 异常类型：数值跳变、持续偏离、周期性波动、噪声增大、数据缺失、尖峰脉冲

## 常见模式
- 磁力计异常+姿态偏差 → 磁力矩器故障/外部磁干扰
- 光电二极管异常+单轴偏离 → 太阳敏感器遮挡/姿态控制问题
- 多通道同时异常 → 电源/总线故障/地磁暴/进出阴影
- 周期性异常 → 轨道周期相关（地影/温度循环）

## 输出硬约束（违反则不合格）
1. 固定结构：【通道定位】→【异常类型】→【可能原因】→【影响评估】→【建议措施】→【紧急程度】→【来源】
2. 【可能原因】最多3条，降序，每条≤30字
3. 【影响评估】【建议措施】各最多2条，每条≤30字
4. 【紧急程度】仅：低/中/高/紧急
5. 【来源】格式：[文档名 p.页码]，无依据写"知识库未覆盖"
6. 禁止：寒暄、解释性废话、"综上所述"、markdown标题、emoji
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

        self.system_prompt = self.prompt_config.get("system_prompt", self.default_system_prompt)
        self.user_template = self.prompt_config.get("user_template", self.default_user_template)
        self.citation_format = self.prompt_config.get("citation_format", self.default_citation_format)

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
            f"问题: {query}\n\n"
            f"参考知识:\n{context}\n\n"
            "要求：\n"
            "1. 直接回答，不解释问题背景\n"
            "2. 分点列出，每点一行，每点≤30字\n"
            "3. 标注来源，格式：[文档名 p.页码]\n"
            "4. 知识不足写\"知识库未覆盖\"\n"
            "5. 禁止寒暄、总结套话"
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
