"""PII Guard - 个人身份信息防护模块

提供 PII（Personal Identifiable Information）检测和处理功能。

支持的 PII 类型:
- EMAIL: 邮箱地址
- PHONE: 电话号码（预留）
- ID_CARD: 身份证号（预留）
- BANK_CARD: 银行卡号（预留）

支持的处理策略:
- REDACT: 替换为占位符 [REDACTED_XXX]
- MASK: 部分遮蔽（如 t***@example.com）
- BLOCK: 拒绝处理包含 PII 的输入
- LOG: 仅记录日志，不做处理
"""
import re
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass


class PIIType(Enum):
    """PII 类型枚举"""
    EMAIL = "email"
    PHONE = "phone"
    ID_CARD = "id_card"
    BANK_CARD = "bank_card"


class PIIStrategy(Enum):
    """PII 处理策略"""
    REDACT = "redact"      # 完全替换为占位符
    MASK = "mask"          # 部分遮蔽
    BLOCK = "block"        # 拒绝处理
    LOG = "log"            # 仅记录


@dataclass
class PIIMatch:
    """PII 匹配结果"""
    pii_type: PIIType
    original: str
    start: int
    end: int
    redacted: Optional[str] = None


class PIIGuard:
    """PII 防护器

    检测和处理文本中的个人身份信息。

    使用示例:
    ```python
    guard = PIIGuard(strategy=PIIStrategy.REDACT)

    # 检测 PII
    matches = guard.detect("请发送到 test@example.com")

    # 处理 PII
    result = guard.process("请发送到 test@example.com")
    # result.processed_text == "请发送到 [REDACTED_EMAIL]"
    ```
    """

    # 正则表达式模式
    PATTERNS: Dict[PIIType, str] = {
        # 邮箱地址模式（支持中文语境，不要求单词边界）
        PIIType.EMAIL: r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        # 中国手机号码模式（预留）
        PIIType.PHONE: r'\b1[3-9]\d{9}\b',
        # 中国身份证号模式（预留）
        PIIType.ID_CARD: r'\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b',
        # 银行卡号模式（预留，16-19位数字）
        PIIType.BANK_CARD: r'\b(?:\d{4}[\s-]?){3,4}\d{1,4}\b',
    }

    # 占位符模板
    REDACT_TEMPLATES: Dict[PIIType, str] = {
        PIIType.EMAIL: "[REDACTED_EMAIL]",
        PIIType.PHONE: "[REDACTED_PHONE]",
        PIIType.ID_CARD: "[REDACTED_ID]",
        PIIType.BANK_CARD: "[REDACTED_CARD]",
    }

    def __init__(
        self,
        strategy: PIIStrategy = PIIStrategy.REDACT,
        enabled_types: Optional[List[PIIType]] = None,
        custom_patterns: Optional[Dict[PIIType, str]] = None,
    ):
        """初始化 PII 防护器

        Args:
            strategy: 处理策略，默认为 REDACT
            enabled_types: 启用的 PII 类型，默认仅启用 EMAIL
            custom_patterns: 自定义正则模式
        """
        self.strategy = strategy
        self.enabled_types = enabled_types or [PIIType.EMAIL]

        # 合并自定义模式
        self.patterns = self.PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # 编译正则表达式
        self._compiled_patterns: Dict[PIIType, re.Pattern] = {
            pii_type: re.compile(pattern, re.IGNORECASE)
            for pii_type, pattern in self.patterns.items()
            if pii_type in self.enabled_types
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """检测文本中的 PII

        Args:
            text: 待检测文本

        Returns:
            PIIMatch 列表
        """
        matches = []

        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        # 按位置排序
        matches.sort(key=lambda m: m.start)
        return matches

    def has_pii(self, text: str) -> bool:
        """检查文本是否包含 PII

        Args:
            text: 待检测文本

        Returns:
            True 如果包含 PII
        """
        for pattern in self._compiled_patterns.values():
            if pattern.search(text):
                return True
        return False

    def _redact(self, text: str, matches: List[PIIMatch]) -> str:
        """使用 REDACT 策略处理文本

        Args:
            text: 原始文本
            matches: PII 匹配列表

        Returns:
            处理后的文本
        """
        if not matches:
            return text

        result = []
        last_end = 0

        for match in matches:
            # 添加 PII 之前的文本
            result.append(text[last_end:match.start])
            # 添加占位符
            placeholder = self.REDACT_TEMPLATES.get(
                match.pii_type,
                f"[REDACTED_{match.pii_type.value.upper()}]"
            )
            result.append(placeholder)
            match.redacted = placeholder
            last_end = match.end

        # 添加最后一个 PII 之后的文本
        result.append(text[last_end:])

        return "".join(result)

    def _mask(self, text: str, matches: List[PIIMatch]) -> str:
        """使用 MASK 策略处理文本

        Args:
            text: 原始文本
            matches: PII 匹配列表

        Returns:
            处理后的文本
        """
        if not matches:
            return text

        result = []
        last_end = 0

        for match in matches:
            result.append(text[last_end:match.start])

            # 根据类型进行不同的遮蔽
            if match.pii_type == PIIType.EMAIL:
                # 邮箱: t***@example.com
                masked = self._mask_email(match.original)
            elif match.pii_type == PIIType.PHONE:
                # 手机号: 138****1234
                masked = self._mask_phone(match.original)
            else:
                # 默认: 保留首尾，中间用 * 替换
                masked = self._mask_default(match.original)

            result.append(masked)
            match.redacted = masked
            last_end = match.end

        result.append(text[last_end:])
        return "".join(result)

    def _mask_email(self, email: str) -> str:
        """遮蔽邮箱地址

        Args:
            email: 邮箱地址

        Returns:
            遮蔽后的邮箱 (如 t***@example.com)
        """
        if "@" not in email:
            return self._mask_default(email)

        local, domain = email.split("@", 1)
        if len(local) <= 1:
            masked_local = local + "***"
        else:
            masked_local = local[0] + "***"

        return f"{masked_local}@{domain}"

    def _mask_phone(self, phone: str) -> str:
        """遮蔽手机号

        Args:
            phone: 手机号

        Returns:
            遮蔽后的手机号 (如 138****1234)
        """
        if len(phone) < 7:
            return self._mask_default(phone)
        return phone[:3] + "****" + phone[-4:]

    def _mask_default(self, text: str) -> str:
        """默认遮蔽方式

        Args:
            text: 文本

        Returns:
            遮蔽后的文本
        """
        length = len(text)
        if length <= 2:
            return "*" * length
        return text[0] + "*" * (length - 2) + text[-1]

    def process(self, text: str) -> "PIIProcessResult":
        """处理文本中的 PII

        根据配置的策略处理文本中检测到的 PII。

        Args:
            text: 待处理文本

        Returns:
            PIIProcessResult 包含处理结果

        Raises:
            PIIBlockedError: 当策略为 BLOCK 且检测到 PII 时
        """
        matches = self.detect(text)

        if not matches:
            return PIIProcessResult(
                original_text=text,
                processed_text=text,
                has_pii=False,
                matches=[],
                strategy=self.strategy,
            )

        if self.strategy == PIIStrategy.BLOCK:
            raise PIIBlockedError(
                f"检测到 {len(matches)} 个 PII，请移除后重试",
                matches=matches,
            )

        if self.strategy == PIIStrategy.REDACT:
            processed_text = self._redact(text, matches)
        elif self.strategy == PIIStrategy.MASK:
            processed_text = self._mask(text, matches)
        else:  # LOG strategy
            processed_text = text

        return PIIProcessResult(
            original_text=text,
            processed_text=processed_text,
            has_pii=True,
            matches=matches,
            strategy=self.strategy,
        )


@dataclass
class PIIProcessResult:
    """PII 处理结果"""
    original_text: str
    processed_text: str
    has_pii: bool
    matches: List[PIIMatch]
    strategy: PIIStrategy

    def __str__(self) -> str:
        if not self.has_pii:
            return f"PIIProcessResult(has_pii=False)"
        return (
            f"PIIProcessResult(has_pii=True, "
            f"count={len(self.matches)}, "
            f"strategy={self.strategy.value})"
        )


class PIIBlockedError(Exception):
    """PII 阻止异常

    当策略为 BLOCK 且检测到 PII 时抛出
    """

    def __init__(self, message: str, matches: List[PIIMatch]):
        super().__init__(message)
        self.matches = matches
