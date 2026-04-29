"""
Prompt 模板模块
包含卫星异常诊断专用的提示词模板
"""

from typing import Dict, Any, Optional
import yaml
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplates:
    """
    Prompt 模板管理器
    
    包含功能：
    1. 系统提示词模板
    2. 用户提示词模板
    3. 来源标注格式
    4. 模板变量替换
    
    示例：
        >>> templates = PromptTemplates(config_path="configs/rag_config.yaml")
        >>> system_prompt = templates.get_system_prompt()
        >>> user_prompt = templates.get_user_prompt(query="卫星异常", context="...")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 Prompt 模板管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.prompt_config = self.config.get("prompt", {})
        
        # 默认系统提示词
        self.default_system_prompt = """你是一个卫星异常诊断专家，专门分析OPS-SAT微小卫星的遥测数据异常。
请基于提供的卫星领域知识，给出专业、准确的异常原因分析。

## OPS-SAT 遥测通道速查
9个遥测通道分为两类：
- **磁力计(强通道)**：CADC0872=磁力计X轴, CADC0873=磁力计Y轴, CADC0874=磁力计Z轴
  - 异常含义：磁干扰、磁力矩器故障、姿态控制异常
- **光电二极管(弱通道)**：CADC0884/0886/0888/0890/0892/0894 = 光电二极管1-6角度
  - 异常含义：姿态偏差、传感器遮挡、光照条件变化

## 异常类型分类
1. Unusual shapes(异常形状) 2. Peaks(尖峰) 3. Zero values(零值) 4. Gaps(数据间隙)
CADC0874通道异常以"长数据间隙"为主。

## 回答要求
1. 必须基于提供的知识片段进行分析，不编造
2. 明确标注知识来源（文档名 + 页码）
3. 如果知识片段不足，请说明"现有知识库信息不足"，并给出基于领域常识的推理
4. 使用中文回答，保持专业性和可读性
5. 分析应包含：异常类型判断→可能原因(按可能性排序)→影响范围→建议措施
6. 如果查询涉及具体CADC通道，先说明该通道的物理含义再分析"""
        
        # 默认用户提示词模板
        self.default_user_template = """请分析以下OPS-SAT卫星遥测通道异常的可能原因：

通道ID: {channel_id}
异常类型: {anomaly_type}
异常描述: {anomaly_description}

参考知识：
{context}

请按以下结构给出分析：
1. **通道定位**：该通道属于什么传感器，测量什么物理量
2. **异常类型判断**：属于4种异常类型(异常形状/尖峰/零值/数据间隙)中的哪一种
3. **可能原因**（按可能性从高到低排序）
4. **影响评估**：对卫星系统的潜在影响
5. **建议措施**：排查或处理建议

每个分析点必须标注知识来源。"""
        
        # 来源标注格式
        self.default_citation_format = "【来源：{document_name} 第 {page_number} 页】"
        
        # 从配置加载（如果存在）
        self.system_prompt = self.prompt_config.get("system_prompt", self.default_system_prompt)
        self.user_template = self.prompt_config.get("user_template", self.default_user_template)
        self.citation_format = self.prompt_config.get("citation_format", self.default_citation_format)
        
        logger.info("Prompt 模板管理器初始化完成")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "rag_config.yaml"
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功：{config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件未找到：{config_path}，使用默认配置")
            return {}
        except Exception as e:
            logger.error(f"配置文件加载失败：{e}")
            return {}
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            str: 系统提示词
        """
        return self.system_prompt
    
    def get_user_prompt(
        self, 
        channel_id: str = "",
        anomaly_type: str = "",
        anomaly_description: str = "",
        context: str = "",
        **kwargs
    ) -> str:
        """
        获取用户提示词
        
        Args:
            channel_id: 通道 ID
            anomaly_type: 异常类型
            anomaly_description: 异常描述
            context: 检索到的上下文
            **kwargs: 其他模板变量
        
        Returns:
            str: 用户提示词
        """
        # 准备模板变量
        template_vars = {
            "channel_id": channel_id,
            "anomaly_type": anomaly_type,
            "anomaly_description": anomaly_description,
            "context": context,
            **kwargs
        }
        
        try:
            # 填充模板
            prompt = self.user_template.format(**template_vars)
            return prompt
        except KeyError as e:
            logger.warning(f"模板变量缺失：{e}，使用简化提示词")
            return self._get_simplified_prompt(context, **kwargs)
    
    def _get_simplified_prompt(self, context: str, **kwargs) -> str:
        """
        获取简化提示词（当模板变量缺失时）
        
        Args:
            context: 检索到的上下文
            **kwargs: 其他信息
        
        Returns:
            str: 简化提示词
        """
        prompt = f"""请基于以下卫星领域知识，分析可能的异常原因：

参考知识：
{context}

"""
        
        # 添加其他信息
        if kwargs:
            for key, value in kwargs.items():
                if value:
                    prompt += f"{key}: {value}\n"
        
        prompt += "\n请给出专业分析，并标注知识来源。"
        
        return prompt
    
    def get_rag_prompt(
        self, 
        query: str, 
        context: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        获取完整的 RAG 提示词（系统 + 用户）
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 自定义系统提示词（可选）
        
        Returns:
            Dict[str, str]: 包含 system 和 user 的提示词字典
        """
        # 使用自定义或默认系统提示词
        system = system_prompt or self.system_prompt
        
        # 构建用户提示词
        user = f"""请基于以下卫星领域知识，回答用户的问题：

参考知识：
{context}

用户问题：{query}

请给出专业、准确的回答，并标注知识来源。如果知识库中没有相关信息，请说明"现有知识库信息不足"。"""
        
        return {
            "system": system,
            "user": user
        }
    
    def get_anomaly_analysis_prompt(
        self,
        channel_id: str,
        anomaly_type: str,
        anomaly_description: str,
        context: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        获取异常分析专用提示词
        
        Args:
            channel_id: 通道 ID
            anomaly_type: 异常类型
            anomaly_description: 异常描述
            context: 检索到的上下文
            additional_info: 额外信息
        
        Returns:
            Dict[str, str]: 提示词字典
        """
        # 系统提示词
        system = """你是一个专业的OPS-SAT卫星异常诊断专家，具有丰富的遥测数据分析经验。
请基于提供的卫星领域知识，对异常进行深入分析。

## OPS-SAT 遥测通道速查
- **磁力计(强通道)**：CADC0872=磁力计X轴, CADC0873=磁力计Y轴, CADC0874=磁力计Z轴
  - 异常含义：磁干扰、磁力矩器故障、姿态控制异常
- **光电二极管(弱通道)**：CADC0884/0886/0888/0890/0892/0894 = 光电二极管1-6角度
  - 异常含义：姿态偏差、传感器遮挡、光照条件变化

## 异常类型分类
1. Unusual shapes(异常形状) 2. Peaks(尖峰) 3. Zero values(零值) 4. Gaps(数据间隙)

分析要求：
1. **通道定位**：先说明该通道的传感器类型和物理含义
2. **异常分类**：判断属于哪种异常类型
3. **原因分析**：基于知识库，列出可能的异常原因（按可能性排序）
4. **影响评估**：分析该异常可能对卫星系统造成的影响
5. **紧急程度**：评估异常的紧急程度（低/中/高/紧急）
6. **建议措施**：提供具体的排查和处理建议
7. **知识溯源**：明确标注每个分析点的知识来源

请使用专业但易懂的语言，确保分析结果具有可操作性。不编造知识库中没有的信息。"""
        
        # 用户提示词
        user = f"""请分析以下卫星遥测通道异常：

【异常信息】
- 通道ID：{channel_id}
- 异常类型：{anomaly_type}
- 异常描述：{anomaly_description}
"""
        
        # 添加额外信息
        if additional_info:
            user += "\n【额外信息】\n"
            for key, value in additional_info.items():
                user += f"- {key}：{value}\n"
        
        user += f"""
【参考知识】
{context}

请给出详细的异常分析报告，包含原因分析、影响评估、紧急程度和建议措施。"""
        
        return {
            "system": system,
            "user": user
        }
    
    def get_comparison_prompt(
        self,
        query: str,
        context: str,
        comparison_type: str = "similar"
    ) -> Dict[str, str]:
        """
        获取对比分析提示词
        
        Args:
            query: 查询内容
            context: 检索到的上下文
            comparison_type: 对比类型（similar/different/evolution）
        
        Returns:
            Dict[str, str]: 提示词字典
        """
        # 系统提示词
        system = """你是一个卫星系统专家，擅长对比分析不同技术方案、异常模式或系统特性。
请基于提供的知识，进行专业、客观的对比分析。"""
        
        # 根据对比类型构建用户提示词
        if comparison_type == "similar":
            user = f"""请对比分析以下内容的相似之处：

查询内容：{query}

参考知识：
{context}

请从技术原理、应用场景、优缺点等方面进行对比分析。"""
        
        elif comparison_type == "different":
            user = f"""请对比分析以下内容的差异：

查询内容：{query}

参考知识：
{context}

请从技术原理、性能指标、适用场景等方面进行对比分析。"""
        
        elif comparison_type == "evolution":
            user = f"""请分析以下技术的发展演变：

查询内容：{query}

参考知识：
{context}

请从技术发展、性能提升、应用扩展等方面进行分析。"""
        
        else:
            user = f"""请基于以下知识，分析相关内容：

查询内容：{query}

参考知识：
{context}

请给出专业分析。"""
        
        return {
            "system": system,
            "user": user
        }
    
    def format_citation(self, document_name: str, page_number: int) -> str:
        """
        格式化来源标注
        
        Args:
            document_name: 文档名称
            page_number: 页码
        
        Returns:
            str: 格式化的来源标注
        """
        return self.citation_format.format(
            document_name=document_name,
            page_number=page_number
        )
    
    def update_templates(self, **kwargs):
        """
        更新模板配置
        
        Args:
            **kwargs: 模板配置
        """
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]
        
        if "user_template" in kwargs:
            self.user_template = kwargs["user_template"]
        
        if "citation_format" in kwargs:
            self.citation_format = kwargs["citation_format"]
        
        logger.info("模板配置已更新")
    
    def get_all_templates(self) -> Dict[str, str]:
        """
        获取所有模板
        
        Returns:
            Dict[str, str]: 所有模板
        """
        return {
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "citation_format": self.citation_format
        }
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"PromptTemplates(templates={len(self.get_all_templates())})"


# 全局实例（单例模式）
_global_templates = None

def get_prompt_templates(config_path: Optional[str] = None) -> PromptTemplates:
    """
    获取全局 Prompt 模板实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        PromptTemplates: Prompt 模板实例
    """
    global _global_templates
    if _global_templates is None:
        _global_templates = PromptTemplates(config_path)
    return _global_templates


def get_system_prompt() -> str:
    """
    快速获取系统提示词
    
    Returns:
        str: 系统提示词
    """
    templates = get_prompt_templates()
    return templates.get_system_prompt()


def get_user_prompt(**kwargs) -> str:
    """
    快速获取用户提示词
    
    Args:
        **kwargs: 模板变量
    
    Returns:
        str: 用户提示词
    """
    templates = get_prompt_templates()
    return templates.get_user_prompt(**kwargs)


if __name__ == "__main__":
    """模块测试"""
    print("=== Prompt 模板测试 ===")
    
    # 测试模板管理器
    templates = PromptTemplates()
    print(f"模板管理器：{templates}")
    
    # 测试系统提示词
    system_prompt = templates.get_system_prompt()
    print(f"\n系统提示词：\n{system_prompt[:200]}...")
    
    # 测试用户提示词
    user_prompt = templates.get_user_prompt(
        channel_id="CADC0874",
        anomaly_type="数值异常",
        anomaly_description="遥测值超出正常范围",
        context="卫星遥测数据..."
    )
    print(f"\n用户提示词：\n{user_prompt[:200]}...")
    
    # 测试来源标注
    citation = templates.format_citation("NASA SOA 2024", 42)
    print(f"\n来源标注：{citation}")
    
    print("\n测试完成！")
