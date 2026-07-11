"""add composite index on property latitude/longitude for radius search

Revision ID: b3f8c2d19a47
Revises: 017f9b3e440c
Create Date: 2026-07-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3f8c2d19a47'
down_revision: Union[str, Sequence[str], None] = '017f9b3e440c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        'ix_properties_latitude_longitude',
        'properties',
        ['latitude', 'longitude'],
        unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_properties_latitude_longitude', table_name='properties')
