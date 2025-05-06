<?php

/**
 * classic_widgets_screen Settings Module
 * Settings > Widget Options :: Pages classic_widgets_screen
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Card Module for Pages classic_widgets_screen Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */

/*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */
if (!function_exists('widgetopts_settings_classic_widgets_screen')) :
	function widgetopts_settings_classic_widgets_screen()
	{
		global $widget_options;
?>
		<li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ($widget_options['classic_widgets_screen'] == 'activate') ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-classic_widgets_screen" data-module-id="classic_widgets_screen">
			<div class="widgetopts-module-card-content">
				<h2><?php _e('Classic Widgets Screen', 'widget-options'); ?></h2>
				<p class="widgetopts-module-desc">
					<?php _e('Revert to Classic Widgets Screen (for wordpress version 5.8 and above).', 'widget-options'); ?>
				</p>
				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if ($widget_options['classic_widgets_screen'] == 'activate') { ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e('Learn More', 'widget-options'); ?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e('Disable', 'widget-options'); ?></button>
					<?php } else { ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e('Learn More', 'widget-options'); ?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e('Enable', 'widget-options'); ?></button>
					<?php } ?>

				</div>
			</div>
			<?php widgetopts_modal_start($widget_options['classic_widgets_screen']); ?>
			<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-admin-generic"></span>
			<h3 class="widgetopts-modal-header"><?php _e('Classic Widgets Screen', 'widget-options'); ?></h3>
			<p>
				<?php _e('Simply toggle this option on to revert to WordPress\' classic widget editor.', 'widget-options'); ?>
			</p>
			<p class="widgetopts-settings-section">
				<?php _e('No additional settings available.', 'widget-options'); ?>
			</p>
			<?php widgetopts_modal_end($widget_options['classic_widgets_screen']); ?>
		</li>
<?php
	}
	add_action('widgetopts_module_cards', 'widgetopts_settings_classic_widgets_screen', 10);
endif;
?>