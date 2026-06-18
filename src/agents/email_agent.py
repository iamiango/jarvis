"""Email Agent - 邮件服务 Agent

负责邮件发送功能。
包含 EmailSkill。
支持用户偏好检测和保存。
"""
from typing import Any, Dict, List, Optional
import argparse
import asyncio
import re

from .base import BaseAgent
from ..a2a import AgentSkill, Task, TaskState
from ..skills.email import EmailSkill
from ..skills.user_preference import SaveUserPreferenceSkill
from ..storage.checkpoint import CheckpointManager
from ..config import postgres_config


# 邮件偏好检测规则
EMAIL_PREFERENCE_PATTERNS = {
    "language": {
        "zh": ["中文", "简体", "汉语"],
        "en": ["英文", "English", "英语"],
    },
    "format": {
        "html": ["HTML", "富文本", "格式化"],
        "plain": ["纯文本", "简洁", "plain"],
    },
    "signature_style": {
        "formal": ["正式", "商务", "专业"],
        "casual": ["随意", "简单", "轻松"],
    },
    "notification_frequency": {
        "immediate": ["立即", "实时", "马上"],
        "daily": ["每天", "日报", "每日"],
        "weekly": ["每周", "周报"],
    },
}


class EmailAgent(BaseAgent):
    """Email Agent - 邮件服务专家

    技能:
    - send_email: 发送邮件
    - save_user_preference: 保存用户偏好
    """

    name = "email_agent"
    description = "邮件服务 Agent - 提供邮件发送功能，支持文本、HTML 格式和附件"
    version = "1.0.0"

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager
        # 注册邮件技能
        self._email_skill = EmailSkill(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
        )
        self._save_preference_skill = SaveUserPreferenceSkill(checkpoint_manager)

        self.register_skill(self._email_skill)
        self.register_skill(self._save_preference_skill)

    def get_skills(self) -> List[AgentSkill]:
        """获取 Agent 技能列表"""
        return [
            AgentSkill(
                id="send_email",
                name="发送邮件",
                description="发送邮件到指定地址，支持文本、HTML 格式和附件",
                tags=["邮件", "通知", "发送"],
                examples=[
                    "发送邮件到 test@example.com",
                    "把报告发送到 user@company.com",
                    "发送邮件，主题是'周报'，内容是...",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="save_user_preference",
                name="保存用户偏好",
                description="保存用户的邮件偏好设置到长期记忆",
                tags=["偏好", "设置", "记忆"],
                examples=[
                    "记住我喜欢中文邮件",
                    "我偏好HTML格式",
                ],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ]

    async def process_task(self, task: Task) -> Task:
        """处理邮件发送任务

        从用户消息中解析邮件参数并发送。
        同时检测并保存用户偏好。
        """
        user_text = self._extract_text_from_task(task)
        user_id = task.metadata.get("user_id", "default") if task.metadata else "default"

        if not user_text:
            return self._create_text_response(
                task,
                "请提供邮件信息，例如：发送邮件，主题是'测试'，内容是'这是一封测试邮件'",
                TaskState.INPUT_REQUIRED,
            )

        # 检测并保存用户偏好
        detected_prefs = self._detect_preferences(user_text)
        if detected_prefs and self._checkpoint_manager:
            await self._save_detected_preferences(user_id, detected_prefs)

        # 获取用户偏好以应用到邮件发送
        email_prefs = await self._get_user_email_preferences(user_id)

        # 解析邮件参数
        params = self._parse_email_params(user_text)

        # 应用用户偏好到邮件参数
        params = self._apply_preferences_to_params(params, email_prefs)

        # 收件人可选（使用默认收件人）
        recipient = params.get("to") or self._email_skill.default_to

        if not params.get("subject"):
            return self._create_text_response(
                task,
                "请提供邮件主题",
                TaskState.INPUT_REQUIRED,
            )

        if not params.get("body"):
            return self._create_text_response(
                task,
                "请提供邮件内容",
                TaskState.INPUT_REQUIRED,
            )

        try:
            # 发送邮件（to 可为空，将使用默认收件人）
            result = await self._email_skill.execute(
                to=params.get("to"),  # 可为 None
                subject=params["subject"],
                body=params["body"],
                html=params.get("html", False),
            )

            if result.success:
                return self._create_text_response(
                    task,
                    f"邮件已成功发送到 {recipient}",
                )
            else:
                return self._create_text_response(
                    task,
                    f"邮件发送失败: {result.error}",
                    TaskState.FAILED,
                )

        except Exception as e:
            return self._create_text_response(
                task,
                f"邮件发送失败: {str(e)}",
                TaskState.FAILED,
            )

    def _detect_preferences(self, text: str) -> List[Dict[str, Any]]:
        """基于规则检测用户偏好

        检测模式:
        - language: 中文/英文
        - format: HTML/纯文本
        - signature_style: 正式/随意
        - notification_frequency: 立即/每天/每周

        Args:
            text: 用户输入文本

        Returns:
            检测到的偏好列表
        """
        detected = []

        for pref_key, value_patterns in EMAIL_PREFERENCE_PATTERNS.items():
            for value, keywords in value_patterns.items():
                for keyword in keywords:
                    if keyword.lower() in text.lower():
                        confidence = min(0.7 + len(keyword) * 0.05, 1.0)
                        detected.append({
                            "key": pref_key,
                            "value": value,
                            "confidence": confidence,
                            "matched_keyword": keyword,
                        })
                        break

        # 去重
        unique_prefs = {}
        for pref in detected:
            key = pref["key"]
            if key not in unique_prefs or pref["confidence"] > unique_prefs[key]["confidence"]:
                unique_prefs[key] = pref

        return list(unique_prefs.values())

    async def _save_detected_preferences(self, user_id: str, preferences: List[Dict[str, Any]]) -> None:
        """保存检测到的偏好"""
        for pref in preferences:
            try:
                await self._save_preference_skill.execute(
                    user_id=user_id,
                    category="email",
                    key=pref["key"],
                    value=pref["value"],
                    confidence=pref["confidence"],
                    source="inferred",
                    checkpoint_manager=self._checkpoint_manager,
                )
                print(f"💾 已保存用户偏好: {pref['key']}={pref['value']} (置信度: {pref['confidence']:.2f}, 关键词: {pref['matched_keyword']})")
            except Exception as e:
                print(f"⚠️ 保存偏好失败: {e}")

    async def _get_user_email_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户邮件偏好"""
        if not self._checkpoint_manager:
            return {}

        try:
            prefs = await self._checkpoint_manager.get_preferences(user_id, "email")
            return {p["key"]: p["value"] for p in prefs}
        except Exception as e:
            print(f"⚠️ 获取用户偏好失败: {e}")
            return {}

    def _apply_preferences_to_params(self, params: Dict[str, Any], prefs: Dict[str, Any]) -> Dict[str, Any]:
        """应用用户偏好到邮件参数

        Args:
            params: 解析的邮件参数
            prefs: 用户偏好

        Returns:
            应用偏好后的参数
        """
        # 如果用户偏好 HTML 格式，且参数中未指定
        if prefs.get("format") == "html" and "html" not in params:
            params["html"] = True

        return params

    def _parse_email_params(self, text: str) -> dict:
        """从文本中解析邮件参数

        支持的格式:
        - 发送邮件到 xxx@xxx.com，主题是'xxx'，内容是'xxx'
        - 发送到: xxx@xxx.com, 主题: xxx, 内容: xxx
        - JSON 格式: {"to": "xxx", "subject": "xxx", "body": "xxx"}
        """
        params = {}

        # 尝试解析邮箱地址
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        if emails:
            params["to"] = emails[0]

        # 尝试解析主题 - 使用更宽松的模式
        subject_patterns = [
            r"主题[是为：:]\s*'([^']+)'",  # 单引号包裹
            r'主题[是为：:]\s*"([^"]+)"',  # 双引号包裹
            r'主题[是为：:]\s*([^，,。\n\'\"]+)',  # 无引号
            r"subject[:\s]+'([^']+)'",
            r'subject[:\s]+"([^"]+)"',
            r'subject[:\s]+([^,\n]+)',
        ]
        for pattern in subject_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                params["subject"] = match.group(1).strip()
                break

        # 尝试解析正文 - 对于长文本，提取"内容:"之后的所有内容
        # 首先尝试找到"内容:"的位置
        body_start_patterns = [
            r"内容[是为：:]\s*'",  # 内容: '
            r'内容[是为：:]\s*"',  # 内容: "
            r'内容[是为：:]\s*',   # 内容:
        ]

        body_found = False
        for pattern in body_start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.end()
                # 获取开始字符（引号类型）
                start_char = text[match.end() - 1] if match.end() > 0 else ''

                if start_char in ["'", '"']:
                    # 如果是引号开始，找到对应的结束引号（从末尾找）
                    # 由于内容可能包含引号，我们从末尾向前找最后一个引号
                    remaining = text[start_pos:]
                    # 找最后一个对应的引号
                    last_quote_pos = remaining.rfind(start_char)
                    if last_quote_pos > 0:
                        params["body"] = remaining[:last_quote_pos]
                    else:
                        params["body"] = remaining
                else:
                    # 无引号，取剩余所有内容
                    params["body"] = text[start_pos:].strip()

                body_found = True
                break

        # 如果上面的方法没找到，尝试更简单的模式
        if not body_found:
            simple_patterns = [
                r'body[:\s]+(.+)',
            ]
            for pattern in simple_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    params["body"] = match.group(1).strip()
                    break

        # 检查是否包含 HTML 标记
        if params.get("body"):
            if "<html" in params["body"].lower() or "<p>" in params["body"].lower():
                params["html"] = True

        return params

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
    ) -> dict:
        """直接发送邮件（供其他 Agent 调用）

        Args:
            to: 收件人地址
            subject: 邮件主题
            body: 邮件正文
            html: 是否为 HTML 格式

        Returns:
            包含发送结果的字典
        """
        result = await self._email_skill.execute(
            to=to,
            subject=subject,
            body=body,
            html=html,
        )
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
        }


def main():
    """Email Agent 独立启动入口"""
    parser = argparse.ArgumentParser(description="Email Agent - 邮件发送服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8002, help="服务端口")
    parser.add_argument("--smtp-host", help="SMTP 服务器地址")
    parser.add_argument("--smtp-port", type=int, help="SMTP 端口")
    parser.add_argument("--smtp-user", help="SMTP 用户名")
    parser.add_argument("--smtp-password", help="SMTP 密码")
    args = parser.parse_args()

    print(f"启动 Email Agent...")
    print(f"  - 地址: http://{args.host}:{args.port}")
    print(f"  - Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")

    # 创建 CheckpointManager 以支持用户偏好存储
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(postgres_config.connection_string)
        print("  - ✅ 用户偏好存储已配置 (PostgreSQL)")
    except Exception as e:
        print(f"  - ⚠️ 用户偏好存储未启用: {e}")
        checkpoint_manager = None

    agent = EmailAgent(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password,
        checkpoint_manager=checkpoint_manager,
    )
    agent.start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
